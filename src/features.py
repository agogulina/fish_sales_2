"""
Признаки для моделей: лаги/скользящие/сезонность, парсер веса из наименований,
словарь продуктов и (опционально) семантические эмбеддинги e5.
"""
import re
import numpy as np
import pandas as pd

from config import COL_SKU, COL_CLIENT, COL_NAME, E5_MODEL, EMB_DIM, RANDOM_STATE


# ----------------------------------------------------------------- временные признаки
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги (1/2/3/6/12), скользящие средние/станд., сезонность. Вход: панель с 'qty','month'."""
    out = df.copy().sort_values([COL_SKU, COL_CLIENT, "month"])
    out["month_num"] = out["month"].dt.month.astype(int)
    out["quarter"]   = out["month"].dt.quarter.astype(int)
    out["m_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["m_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    grp = out.groupby([COL_SKU, COL_CLIENT], sort=False)
    for lag in [1, 2, 3, 6, 12]:
        out[f"lag_qty_{lag}"] = grp["qty"].shift(lag)
    out["roll3_qty_mean"] = grp["qty"].shift(1).transform(lambda x: x.rolling(3, min_periods=1).mean())
    out["roll6_qty_mean"] = grp["qty"].shift(1).transform(lambda x: x.rolling(6, min_periods=1).mean())
    out["roll3_qty_std"]  = grp["qty"].shift(1).transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    out["is_high_season"] = out["month_num"].isin([11, 12, 1, 2]).astype(int)
    lag_cols = [c for c in out.columns if c.startswith("lag_") or c.startswith("roll")]
    out[lag_cols] = out[lag_cols].fillna(0.0)
    out["y"] = np.log1p(out["qty"].clip(lower=0))
    return out


SEASON_COLS = ["m_sin", "m_cos", "is_high_season", "month_num", "quarter"]
LAG_COLS    = ["lag_qty_1", "lag_qty_2", "lag_qty_3", "lag_qty_6", "lag_qty_12",
               "roll3_qty_mean", "roll6_qty_mean", "roll3_qty_std"]


# ----------------------------------------------------------------- вес из наименования
def parse_weight_g(name):
    """Возвращает (способ, вес_в_граммах|nan). Упаковочные дроби '1/6' игнорируются."""
    t = str(name).lower().replace("ё", "е").replace(",", ".")
    t = re.sub(r"\b\d+\s*/\s*\d+\b", " ", t)          # убрать '1/6','1/8' (штук в коробе)
    if re.search(r"\bвес\b|весов", t):
        return ("весовой", np.nan)
    m = re.search(r"(\d+(?:\.\d+)?)\s*кг", t)
    if m:
        return ("кг", float(m.group(1)) * 1000)
    m = re.search(r"(\d+(?:\.\d+)?)\s*гр?\b", t)
    if m:
        return ("г", float(m.group(1)))
    m = re.search(r"(?<![\d.])(\d\.\d+)(?![\d])", t)   # голое десятичное -> кг
    if m:
        v = float(m.group(1))
        return ("голое-кг", v * 1000) if 0.02 <= v <= 5 else ("нет", np.nan)
    m = re.search(r"(?<![\d.])(\d{2,4})(?![\d./])", t) # голое целое -> граммы
    if m:
        v = float(m.group(1))
        return ("голое-г", v) if 50 <= v <= 2000 else ("нет", np.nan)
    return ("нет", np.nan)


# ----------------------------------------------------------------- словарь продуктов
PRODUCT_HEADS = {
    "имбирь": "имбирь", "анчоус": "анчоус", "хамса": "анчоус", "килька": "килька", "икра": "икра",
    "капуста": "морская капуста", "скумбрия": "скумбрия", "сельдь": "сельдь", "семга": "семга",
    "сёмга": "семга", "форель": "форель", "горбуша": "горбуша", "салака": "салака", "мойва": "мойва",
    "ставрида": "ставрида", "креветка": "креветка", "паштет": "паштет", "спред": "спред",
    "пресерв": "пресервы", "морепродукт": "морепродукты", "масло": "масло", "мидии": "мидии",
    "кальмар": "кальмар", "осьминог": "осьминог", "тунец": "тунец", "лосось": "лосось",
    "кета": "кета", "сардина": "сардина", "масляная": "масляная рыба", "масляной": "масляная рыба",
    "треска": "треска", "минтай": "минтай", "краб": "краб", "путассу": "путассу",
    "сайра": "сайра", "салат": "салат",
}


def product_head(name):
    """Грубый «продукт» из наименования (для сопоставления цен и группировки)."""
    t = str(name).lower().replace("ё", "е")
    for k, c in PRODUCT_HEADS.items():
        if k in t:
            return c
    return None


# ----------------------------------------------------------------- эмбеддинги e5 (опционально)
def e5_embeddings(catalog: pd.DataFrame, dim=EMB_DIM):
    """
    Эмбеддинги наименований моделью e5 -> PCA до `dim` признаков.
    Возвращает DataFrame [COL_SKU, emb_0..emb_{dim-1}] или None, если модель недоступна.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        names = catalog[[COL_SKU, COL_NAME]].dropna().drop_duplicates(COL_SKU)
        model = SentenceTransformer(E5_MODEL, device=device)
        E = model.encode(["passage: " + str(n) for n in names[COL_NAME]],
                         batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        k = min(dim, E.shape[0], E.shape[1])
        Ep = PCA(n_components=k, random_state=RANDOM_STATE).fit_transform(E)
        cols = [f"emb_{i}" for i in range(k)]
        out = pd.DataFrame(Ep, columns=cols)
        out[COL_SKU] = names[COL_SKU].values
        return out
    except Exception as e:                              # noqa
        print("e5 недоступна -> работаем без семантических признаков:", repr(e))
        return None
