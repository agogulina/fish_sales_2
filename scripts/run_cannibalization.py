
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config import COL_SKU, COL_GROUP
from src.data import build_panel, load_catalog
from src.cannibalization import (build_substitute_pressure, within_bootstrap,
                                  fe_cluster_se, dml_cannibalization)

if __name__ == "__main__":
    panel = build_panel()
    catalog = load_catalog()
    groups = catalog.set_index(COL_SKU)[COL_GROUP]
    df = build_substitute_pressure(panel, groups)

    c, lo, hi = within_bootstrap(df)
    print(f"[within + бутстрэп]  эффект субститутов = {c:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]")

    fe = fe_cluster_se(df, by_cat_month=False)
    if fe:
        print(f"[FE SKU+месяц]       субституты {fe['subst'][0]:+.4f} {fe['subst'][1]}  | "
              f"собств.скидка {fe['own'][0]:+.4f} (позитивный контроль)")

    th, se, p, lo2, hi2 = dml_cannibalization(df)
    print(f"[Double ML]          эффект субститутов = {th:+.4f}  CI=[{lo2:+.4f},{hi2:+.4f}]")

