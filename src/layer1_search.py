"""
Слой 1 — семантический поиск товаров.
Шесть методов (от наивного к LLM), метрики ранжирования и проверка значимости.
Тяжёлые модели (e5, реранкер, LLM, pymorphy3, rapidfuzz, faiss) подключаются
по возможности; при их отсутствии соответствующие методы пропускаются.
"""
import re, math
import numpy as np
import pandas as pd

from config import (COL_NAME, COL_CAT, COL_GROUP, E5_MODEL, RERANKER_MODEL,
                    K, CAND_POOL, RANDOM_STATE)
from src.features import PRODUCT_HEADS, product_head

# ------------------------------------------------- лексическая нормализация
ABBREV = {"с/с": " слабосоленый ", "п/п": " пряного посола ", "пр/посола": " пряного посола ",
          "пр/п": " пряного посола ", "в/у": " вакуумная упаковка ", "в/м": " в масле ",
          "г/к": " горячего копчения ", "х/к": " холодного копчения "}
STOP = {"кг", "г", "гр", "мл", "л", "шт", "для", "со", "в", "на", "и", "с", "из", "без", "от", "до"}

try:
    import pymorphy3
    from rapidfuzz import fuzz
    _morph = pymorphy3.MorphAnalyzer()
    _LEX_OK = True
except Exception:                                       # noqa
    _morph, fuzz, _LEX_OK = None, None, False


def normalize_name(text: str) -> str:
    """Нижний регистр, раскрытие сокращений, удаление чисел/пунктуации, лемматизация, сортировка токенов."""
    t = str(text).lower().replace("ё", "е")
    for k, v in ABBREV.items():
        t = t.replace(k, v)
    t = re.sub(r"\d+[.,]?\d*", " ", t)       # числа и размеры -> убираем
    t = re.sub(r"[^a-zа-я\s]", " ", t)        # пунктуация, дроби фасовки -> убираем
    toks = [w for w in t.split() if w not in STOP and len(w) > 1]
    if _LEX_OK:
        toks = [_morph.parse(w)[0].normal_form for w in toks]
    return " ".join(sorted(set(toks)))


# ------------------------------------------------- атрибуты (структурный метод)
ATTR_PATTERNS = {
    "обработка": {"маринованный": ["маринован", "маринад"],
                  "копченый": ["копчен", "горячего копчения", "холодного копчения", "г/к", "х/к"],
                  "слабосоленый": ["слабосол", "малосол", "с/с"],
                  "пряного посола": ["пряного посола", "п/п", "пр/посола"],
                  "имитированная": ["имитирован", "аналогов"], "соленый": ["солен"],
                  "вяленый": ["вялен", "сушен"]},
    "форма":    {"филе": ["филе"], "кусок": ["кусок", "куск"], "нарезка": ["нарезк"]},
    "упаковка": {"стекло": ["стекл"], "пластик": ["пластик"], "банка": ["банк"],
                 "тиромат": ["тиромат"], "вакуум": ["вакуум", "в/у"]},
    "цвет":     {"красный": ["красн"], "черный": ["черн"], "розовый": ["розов"], "белый": ["бел"]},
}


def extract_attributes(name: str) -> dict:
    t = str(name).lower().replace("ё", "е")
    a = {"product": product_head(t)}
    for grp, opts in ATTR_PATTERNS.items():
        a[grp] = None
        for val, pats in opts.items():
            if any(p in t for p in pats):
                a[grp] = val; break
    return a


def structured_score(qa: dict, da: dict) -> float:
    if qa.get("product") and da.get("product"):
        if qa["product"] != da["product"]:
            return 0.0
        s = 3.0
    else:
        s = 0.0
    for k in ["обработка", "форма", "упаковка", "цвет"]:
        if qa.get(k) and qa[k] == da.get(k):
            s += 1.0
    return s


# =================================================================
class SearchEngine:
    """Каталог + все методы поиска. Тяжёлые модели грузятся лениво и опционально."""

    def __init__(self, catalog: pd.DataFrame):
        self.cat = catalog.reset_index(drop=True).copy()
        self.cat["norm"]  = self.cat[COL_NAME].apply(normalize_name)
        self.cat["attrs"] = self.cat[COL_NAME].apply(extract_attributes)
        self.ids = self.cat["doc_id"].to_numpy()
        self._embedder = self._index = self._reranker = None
        self._build_category_baseline()

    # ----- 0. baseline по корню слова (категория из наименования)
    def _build_category_baseline(self):
        cats = sorted(self.cat[COL_CAT].dropna().unique().tolist())
        self._categories = cats
        self._cat_norm = {c: normalize_name(c) for c in cats}

        def roots(text, n=4):
            t = re.sub(r"[^a-zа-я\s/]", " ", str(text).lower().replace("ё", "е"))
            toks = [w for w in re.split(r"[\s/]+", t) if len(w) >= 4]
            if _LEX_OK:
                return {_morph.parse(w)[0].normal_form[:n] for w in toks}
            return {w[:n] for w in toks}
        self._roots = roots
        self._cat_roots = {c: roots(c) for c in cats}

        def best_category(text):
            r = roots(text)
            ov = {c: len(r & s) for c, s in self._cat_roots.items()}
            best = max(ov, key=ov.get)
            if ov[best] > 0:
                return best, float(ov[best])
            if _LEX_OK:
                nt = normalize_name(text)
                sims = {c: fuzz.token_set_ratio(nt, self._cat_norm[c]) for c in cats}
                best = max(sims, key=sims.get)
                return best, sims[best] / 100.0
            return best, 0.0
        self._best_category = best_category
        self._prod_cat = {r["doc_id"]: best_category(r[COL_NAME])[0] for _, r in self.cat.iterrows()}
        self._prod_score = {r["doc_id"]: best_category(r[COL_NAME])[1] for _, r in self.cat.iterrows()}

    def retrieve_category_baseline(self, query, **kw):
        qc, _ = self._best_category(query)
        sc = np.array([self._prod_score[d] if self._prod_cat[d] == qc else -1.0 for d in self.ids], float)
        order = np.argsort(-sc)
        return self.ids[order], sc[order]

    # ----- 1. лексический
    def retrieve_lexical(self, query, **kw):
        if not _LEX_OK:
            return self.ids, np.zeros(len(self.ids))
        qn = normalize_name(query)
        sc = self.cat["norm"].apply(lambda d: fuzz.token_set_ratio(qn, d)).to_numpy(float)
        order = np.argsort(-sc)
        return self.ids[order], sc[order]

    # ----- 2. плотный e5 + FAISS
    def _ensure_dense(self):
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            import faiss, torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedder = SentenceTransformer(E5_MODEL, device=dev)
            emb = self._embedder.encode(["passage: " + n for n in self.cat[COL_NAME]],
                                        batch_size=64, normalize_embeddings=True,
                                        convert_to_numpy=True).astype("float32")
            self._index = faiss.IndexFlatIP(emb.shape[1]); self._index.add(emb)
            return True
        except Exception as e:                          # noqa
            print("Плотный поиск недоступен:", repr(e)); return False

    def retrieve_dense(self, query, topn=None, **kw):
        if not self._ensure_dense():
            return self.ids, np.zeros(len(self.ids))
        qv = self._embedder.encode(["query: " + query], normalize_embeddings=True,
                                   convert_to_numpy=True).astype("float32")
        k = len(self.cat) if topn is None else min(topn, len(self.cat))
        sims, idx = self._index.search(qv, k)
        return self.ids[idx[0]], sims[0]

    # ----- 3. e5 + реранкер (лучший метод)
    def _ensure_reranker(self):
        if self._reranker is not None:
            return True
        try:
            from sentence_transformers import CrossEncoder
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._reranker = CrossEncoder(RERANKER_MODEL, device=dev, max_length=256)
            return True
        except Exception as e:                          # noqa
            print("Реранкер недоступен:", repr(e)); return False

    def retrieve_rerank(self, query, candidate_pool=CAND_POOL, **kw):
        if not (self._ensure_dense() and self._ensure_reranker()):
            return self.retrieve_dense(query)
        cand, _ = self.retrieve_dense(query, topn=candidate_pool)
        names = self.cat.set_index("doc_id").loc[list(cand), COL_NAME].tolist()
        ce = np.asarray(self._reranker.predict([[query, nm] for nm in names]), float)
        order = np.argsort(-ce)
        rid, rsc = np.asarray(cand)[order], ce[order]
        seen = set(int(x) for x in cand)
        rest = [d for d in self.ids.tolist() if d not in seen]
        ids = np.concatenate([rid, np.array(rest, dtype=rid.dtype)])
        sc = np.concatenate([rsc, np.full(len(rest), rsc.min() - 1.0)])
        return ids, sc

    # ----- 4. структурный (правила)
    def retrieve_structured(self, query, **kw):
        qa = extract_attributes(query)
        sc = self.cat["attrs"].apply(lambda da: structured_score(qa, da)).to_numpy(float)
        order = np.argsort(-sc)
        return self.ids[order], sc[order]

    # ----- методы как словарь
    def methods(self):
        return {
            "Категорийный baseline": self.retrieve_category_baseline,
            "Лексический":           self.retrieve_lexical,
            "Плотный (e5)":          self.retrieve_dense,
            "e5 + реранкер":         self.retrieve_rerank,
            "Структурный (правила)": self.retrieve_structured,
        }

    # ----- продакшн-функция
    def find_products(self, query, threshold=None, pool=CAND_POOL, topk=None):
        ids, sc = self.retrieve_rerank(query, candidate_pool=pool)
        df = self.cat.set_index("doc_id").loc[list(ids)][[COL_NAME, COL_CAT, COL_GROUP]].copy()
        df["score"] = sc
        if threshold is not None:
            df = df[df["score"] >= threshold]
        return df.head(topk) if topk else df


# =================================================================
# метрики ранжирования
def p_at_k(r, rel, k): return len(set(r[:k]) & rel) / k
def r_at_k(r, rel, k): return len(set(r[:k]) & rel) / len(rel) if rel else 0.0
def rr(r, rel):
    for i, d in enumerate(r, 1):
        if d in rel:
            return 1.0 / i
    return 0.0
def ap(r, rel):
    if not rel:
        return 0.0
    hit, s = 0, 0.0
    for i, d in enumerate(r, 1):
        if d in rel:
            hit += 1; s += hit / i
    return s / len(rel)
def ndcg_at_k(r, rel, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(r[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(engine: "SearchEngine", queries, gold, k=K):
    """queries=[(текст, predicate)], gold={текст:set(doc_id)}. Возвращает (summary, per_query_ap)."""
    methods = engine.methods()
    rows, per_q = [], {m: [] for m in methods}
    for q, _ in queries:
        rel = gold[q]
        for m, fn in methods.items():
            ids, _ = fn(q)
            r = [int(x) for x in ids]
            a = ap(r, rel); per_q[m].append(a)
            rows.append({"метод": m, f"P@{k}": p_at_k(r, rel, k), f"R@{k}": r_at_k(r, rel, k),
                         "AP": a, "RR": rr(r, rel), f"nDCG@{k}": ndcg_at_k(r, rel, k)})
    res = pd.DataFrame(rows)
    summary = (res.groupby("метод")[[f"P@{k}", f"R@{k}", "AP", "RR", f"nDCG@{k}"]]
                  .mean().round(3).rename(columns={"AP": "MAP", "RR": "MRR"})
                  .sort_values("MAP", ascending=False))
    return summary, per_q


def significance(per_query_ap):
    """Тест Фридмана + (если есть scikit_posthocs) Неменьи. Возвращает словарь с результатами."""
    from scipy.stats import friedmanchisquare
    ms = list(per_query_ap)
    AP = np.array([per_query_ap[m] for m in ms])
    out = {}
    stat, p = friedmanchisquare(*[AP[i] for i in range(len(ms))])
    out["friedman_chi2"], out["friedman_p"] = float(stat), float(p)
    ap_df = pd.DataFrame(AP.T, columns=ms)
    out["mean_rank"] = ap_df.rank(axis=1, ascending=False).mean().sort_values()
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(ap_df.values)
        nem.index = ms; nem.columns = ms
        out["nemenyi"] = nem
    except Exception:                                   # noqa
        out["nemenyi"] = None
    return out
