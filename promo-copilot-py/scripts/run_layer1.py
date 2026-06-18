"""Слой 1. Семантический поиск: сравнение методов, метрики, значимость."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import COL_GROUP
from src.data import load_catalog
from src.layer1_search import SearchEngine, evaluate, significance

# Тестовые запросы и эталон (по человеческой разметке Группа new).
# При необходимости отредактируйте список под свои группы.
def g_eq(v):         return lambda g: str(g) == v
def g_contains(s):   return lambda g: s.lower() in str(g).lower()
def g_startswith(p): return lambda g: str(g).startswith(p)

QUERIES = [
    ("маринованный имбирь", g_contains("имбирь")),
    ("анчоусы", g_eq("Анчоусы")),
    ("копчёная скумбрия", g_eq("Копчение Скумбрия")),
    ("копчёная сельдь", g_startswith("Копчение Сельдь")),
    ("сёмга", g_eq("#КР_Семга")),
    ("форель", g_eq("#КР_Форель")),
    ("килька пряного посола", g_eq("Рыба п\\п Килька")),
    ("спред из креветки", g_eq("Спреды Креветка рубленая")),
]

if __name__ == "__main__":
    catalog = load_catalog()
    print("Каталог:", len(catalog), "товаров,", catalog[COL_GROUP].nunique(), "групп")
    eng = SearchEngine(catalog)
    gold = {q: set(catalog.loc[catalog[COL_GROUP].apply(pred), "doc_id"]) for q, pred in QUERIES}
    gold = {q: g for q, g in gold.items() if g}            # выкинуть пустые эталоны
    queries = [(q, p) for q, p in QUERIES if q in gold]
    summary, per_q = evaluate(eng, queries, gold)
    print("\nСводные метрики (макро-среднее):"); print(summary.to_string())
    sig = significance(per_q)
    print(f"\nТест Фридмана: chi2={sig['friedman_chi2']:.3f}, p={sig['friedman_p']:.4f}")
    print("Средний ранг (1=лучший):"); print(sig["mean_rank"].round(3).to_string())
    # демо продакшн-поиска
    print("\nfind_products('копчёная горбуша') -> топ-5:")
    print(eng.find_products("копчёная горбуша", topk=5).to_string())
