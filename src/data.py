"""
Загрузка данных и построение помесячной панели.
Содержит общие помощники, которыми пользуются все слои.
"""
import os, glob
import numpy as np
import pandas as pd

from config import (COL_SKU, COL_NAME, COL_CLIENT, COL_DATE, COL_QTY,
                    COL_CAT, COL_GROUP, COL_TYPE, COL_WSTART, COL_PPLAN,
                    COL_PDISC, OPT_CATS, SALES_CSV, PROMO_SALES_CSV, PROMO_FINAL_CSV)


# ----------------------------------------------------------------- помощники
def safe_num(s: pd.Series) -> pd.Series:
    """Числа в русском формате ('1 234,5') -> float."""
    if s.dtype == "O":
        s = s.astype(str).str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def to_month(dt) -> pd.Series:
    """Любая дата -> первое число месяца."""
    return pd.to_datetime(dt, dayfirst=True, errors="coerce").dt.to_period("M").dt.to_timestamp()


def _mode(s: pd.Series):
    s = s.dropna().astype(str)
    return s.mode().iloc[0] if len(s) else np.nan


def first_existing(patterns):
    for p in patterns:
        for f in glob.glob(p):
            return f
    return None


# ----------------------------------------------------------------- каталог (Слой 1)
def load_catalog(sales_csv=SALES_CSV, promo_sales_csv=PROMO_SALES_CSV) -> pd.DataFrame:
    """Уникальный каталог товаров (SKU + наименование + категория/группа/тип)."""
    frames = []
    for path in [sales_csv, promo_sales_csv]:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            keep = [c for c in [COL_TYPE, COL_CAT, COL_GROUP, COL_NAME, COL_SKU] if c in df.columns]
            frames.append(df[keep])
    if not frames:
        raise FileNotFoundError("Не найдены sales.csv / promo_sales1.csv в папке данных")
    raw = pd.concat(frames, ignore_index=True)
    catalog = (raw.dropna(subset=[COL_NAME, COL_SKU])
                  .groupby(COL_SKU, as_index=False)
                  .agg({COL_NAME: _mode, COL_CAT: _mode, COL_GROUP: _mode, COL_TYPE: _mode}))
    catalog[COL_NAME] = catalog[COL_NAME].astype(str).str.strip()
    catalog = catalog.drop_duplicates(subset=[COL_NAME]).reset_index(drop=True)
    catalog["doc_id"] = np.arange(len(catalog))
    return catalog


# ----------------------------------------------------------------- помесячная панель (Слои 2–3)
def agg_monthly(df: pd.DataFrame, qty_col: str) -> pd.DataFrame:
    """Сумма продаж по SKU×клиент×месяц + наиболее частые категориальные метки."""
    d = df.copy()
    d[COL_DATE] = to_month(d[COL_DATE])
    d[COL_QTY] = safe_num(d[COL_QTY]).fillna(0.0)
    agg = {COL_QTY: "sum"}
    for c in OPT_CATS:
        if c in d.columns:
            agg[c] = _mode
    out = d.groupby([COL_SKU, COL_CLIENT, COL_DATE], as_index=False).agg(agg)
    return out.rename(columns={COL_QTY: qty_col, COL_DATE: "month"})


def load_discounts(promo_final_csv=PROMO_FINAL_CSV) -> pd.DataFrame:
    """Недельные скидки -> помесячный max % скидки (лечение для Слоёв 2–3)."""
    pf = pd.read_csv(promo_final_csv)
    pf[COL_WSTART] = pd.to_datetime(pf[COL_WSTART], dayfirst=True, errors="coerce")
    pf["month"] = pf[COL_WSTART].dt.to_period("M").dt.to_timestamp()
    pf[COL_PDISC] = safe_num(pf[COL_PDISC]).fillna(0.0)
    if COL_PPLAN in pf.columns:
        pf[COL_PPLAN] = safe_num(pf[COL_PPLAN]).fillna(0.0)
    pf[COL_SKU] = pf[COL_SKU].astype(str)
    return pf.groupby([COL_SKU, COL_CLIENT, "month"], as_index=False).agg(
        own_disc=(COL_PDISC, "max"),
        promo_disc_mean=(COL_PDISC, "mean"),
        promo_weeks=(COL_PDISC, lambda x: (x > 0).sum()))


def build_panel(sales_csv=SALES_CSV, promo_final_csv=PROMO_FINAL_CSV) -> pd.DataFrame:
    """Помесячная панель: продажи + категория + глубина скидки."""
    sales = pd.read_csv(sales_csv)
    sm = agg_monthly(sales, "qty")
    sm[COL_SKU] = sm[COL_SKU].astype(str)
    sm["qty"] = sm["qty"].clip(lower=0)                 # возвраты -> 0
    disc = load_discounts(promo_final_csv)
    panel = sm.merge(disc, on=[COL_SKU, COL_CLIENT, "month"], how="left")
    panel["own_disc"] = panel["own_disc"].fillna(0.0)
    panel["promo_disc_mean"] = panel["promo_disc_mean"].fillna(0.0)
    panel["promo_weeks"] = panel["promo_weeks"].fillna(0.0)
    if COL_CAT not in panel.columns:
        panel[COL_CAT] = "unknown"
    panel[COL_CAT] = panel[COL_CAT].fillna("unknown")
    panel["month"] = pd.to_datetime(panel["month"])
    return panel.dropna(subset=["month"]).reset_index(drop=True)
