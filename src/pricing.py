"""
Сопоставление цен: из прайса розничной сети получаем цену за кг по продукту и
переносим её на наши SKU (с нормировкой на вес из наименования).
"""
import numpy as np
import pandas as pd

from config import (COL_SKU, COL_NAME, COL_CAT, COL_GROUP, LENTA_PRICE_XLSX,
                    PRICE_TABLE_XLSX, SALES_CSV)
from src.features import parse_weight_g, product_head


def load_lenta_prices(xlsx=LENTA_PRICE_XLSX, sheet="Задача 1"):
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None)
    df = raw.iloc[5:].copy(); df.columns = range(raw.shape[1])
    pl = df[[1, 57, 58]].copy(); pl.columns = ["name", "вход_шт", "полка_шт"]
    for c in ["вход_шт", "полка_шт"]:
        pl[c] = pd.to_numeric(pl[c], errors="coerce")
    pl = pl.dropna(subset=["name", "вход_шт"])
    return pl[pl["name"].astype(str).str.strip() != "Наименование"]


def price_per_kg_by_product(price_df: pd.DataFrame, lo=100, hi=15000):
    pl = price_df.copy()
    pl["вес_г"] = pl["name"].map(lambda n: parse_weight_g(n)[1])
    pl["продукт"] = pl["name"].map(product_head)
    pl = pl[pl["вес_г"].notna() & (pl["вес_г"] > 0)].copy()
    pl["вход_кг"] = pl["вход_шт"] / (pl["вес_г"] / 1000)
    pl["полка_кг"] = pl["полка_шт"] / (pl["вес_г"] / 1000)
    pl = pl[(pl["вход_кг"].between(lo, hi)) & (pl["полка_кг"].between(lo, hi))]
    byhead = pl.groupby("продукт").agg(вход_кг=("вход_кг", "median"),
                                       полка_кг=("полка_кг", "median"),
                                       n=("name", "size"))
    return byhead, float(pl["вход_кг"].median()), float(pl["полка_кг"].median())


def build_price_table(sales_csv=SALES_CSV, lenta_xlsx=LENTA_PRICE_XLSX, out=PRICE_TABLE_XLSX):
    s = pd.read_csv(sales_csv)
    ours = (s[[COL_SKU, COL_NAME, COL_CAT, COL_GROUP]].dropna(subset=[COL_NAME])
            .drop_duplicates(COL_SKU).reset_index(drop=True))
    ours["вес_способ"], ours["вес_г"] = zip(*ours[COL_NAME].map(parse_weight_g))
    ours["продукт"] = ours[COL_NAME].map(product_head)

    byhead, glob_in, glob_sh = price_per_kg_by_product(load_lenta_prices(lenta_xlsx))

    def assign(r):
        h = r["продукт"]
        if h is not None and h in byhead.index:
            return byhead.loc[h, "вход_кг"], byhead.loc[h, "полка_кг"], "по продукту"
        return glob_in, glob_sh, "запасная медиана"
    ours[["цена_входа_руб_кг", "цена_полки_руб_кг", "метод_цены"]] = ours.apply(
        lambda r: pd.Series(assign(r)), axis=1)
    ours["наценка_%"] = (ours["цена_полки_руб_кг"] / ours["цена_входа_руб_кг"] - 1) * 100
    ours["оценка_цена_входа_шт"] = np.where(ours["вес_г"].notna(),
                                            ours["цена_входа_руб_кг"] * ours["вес_г"] / 1000, np.nan)
    ours["оценка_цена_полки_шт"] = np.where(ours["вес_г"].notna(),
                                            ours["цена_полки_руб_кг"] * ours["вес_г"] / 1000, np.nan)

    out_df = ours.rename(columns={COL_SKU: "Артикул", COL_NAME: "Наименование",
                                  COL_GROUP: "Группа (Слой 1)"})
    cols = ["Артикул", "Наименование", COL_CAT, "Группа (Слой 1)", "продукт", "вес_способ",
            "вес_г", "метод_цены", "цена_входа_руб_кг", "цена_полки_руб_кг", "наценка_%",
            "оценка_цена_входа_шт", "оценка_цена_полки_шт"]
    out_df = out_df[cols]

    m = out_df["метод_цены"] == "по продукту"
    summary = pd.DataFrame({"Показатель": ["Всего SKU", "Цена по продукту", "— % SKU"],
                            "Значение": [len(out_df), int(m.sum()), round(m.mean() * 100, 1)]})
    price_tbl = byhead.reset_index().rename(columns={"продукт": "Продукт"})
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="Сводка", index=False)
            out_df.round(1).to_excel(w, sheet_name="Цены по SKU", index=False)
            price_tbl.round(0).to_excel(w, sheet_name="Цены по продуктам", index=False)
    except Exception as e:                              # noqa
        print("Не удалось сохранить xlsx:", repr(e))
    return out_df, summary
