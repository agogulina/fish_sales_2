"""
Проверка каннибализации.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from config import COL_SKU, COL_CLIENT, COL_CAT, RANDOM_STATE
from src.layer2_causal import dml_plr


def build_substitute_pressure(panel: pd.DataFrame, groups: pd.Series):
    df = panel.copy()
    df["grp"] = df[COL_SKU].map(groups).fillna("—")
    # суммарная и количественная скидка по группе×клиент×месяц
    agg = (df.groupby(["grp", COL_CLIENT, "month"])
             .agg(sum_disc=("own_disc", "sum"), n=("own_disc", "size")).reset_index())
    df = df.merge(agg, on=["grp", COL_CLIENT, "month"], how="left")
    # исключаем собственную скидку -> среднее по «другим» в группе
    df["subst_disc"] = np.where(df["n"] > 1, (df["sum_disc"] - df["own_disc"]) / (df["n"] - 1), 0.0)
    return df


def within_bootstrap(df: pd.DataFrame, seed=RANDOM_STATE, nboot=300):
    d = df.copy()
    d["y"] = np.log1p(d["qty"].clip(lower=0))
    # демировка по SKU×клиент (within)
    grp = d.groupby([COL_SKU, COL_CLIENT])
    d["y_w"] = d["y"] - grp["y"].transform("mean")
    d["x_w"] = d["subst_disc"] - grp["subst_disc"].transform("mean")
    coef = np.polyfit(d["x_w"], d["y_w"], 1)[0]
    rng = np.random.RandomState(seed); uniq = d[COL_SKU].unique()
    idx = {s: d.index[d[COL_SKU] == s].to_numpy() for s in uniq}
    boot = []
    for _ in range(nboot):
        s = rng.choice(uniq, len(uniq), replace=True)
        pos = np.concatenate([idx[x] for x in s])
        sub = d.loc[pos]
        boot.append(np.polyfit(sub["x_w"], sub["y_w"], 1)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(coef), float(lo), float(hi)


def fe_cluster_se(df: pd.DataFrame, by_cat_month=False):
    try:
        import statsmodels.formula.api as smf
    except Exception as e:                              # noqa
        print("statsmodels недоступен:", repr(e)); return None
    d = df.copy()
    d["y"] = np.log1p(d["qty"].clip(lower=0))
    d["mkey"] = d["month"].astype(str)
    if by_cat_month:
        d["fe2"] = d[COL_CAT].astype(str) + "|" + d["mkey"]
        formula = "y ~ subst_disc + own_disc + C(%s) + C(fe2)" % COL_SKU
    else:
        formula = "y ~ subst_disc + own_disc + C(%s) + C(mkey)" % COL_SKU
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d[COL_SKU]})
    return {"subst": (m.params.get("subst_disc"), m.conf_int().loc["subst_disc"].tolist()),
            "own":   (m.params.get("own_disc"),   m.conf_int().loc["own_disc"].tolist())}


def dml_cannibalization(df: pd.DataFrame):
    """Double ML: эффект давления субститутов на лог-продажи, контроли = сезон/категория/собств.скидка."""
    d = df.copy()
    d["y"] = np.log1p(d["qty"].clip(lower=0))
    d["mn"] = d["month"].dt.month
    d["m_sin"] = np.sin(2 * np.pi * d["mn"] / 12); d["m_cos"] = np.cos(2 * np.pi * d["mn"] / 12)
    W = pd.concat([d[["m_sin", "m_cos", "own_disc"]].reset_index(drop=True),
                   pd.get_dummies(d[COL_CAT].astype(str)).reset_index(drop=True)], axis=1).astype(float)
    return dml_plr(d["y"].values, d["subst_disc"].values, W, d[COL_SKU].values)
