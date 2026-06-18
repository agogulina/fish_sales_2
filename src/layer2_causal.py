"""
Слой 2 — причинная оценка отклика спроса на скидку.
Double ML (кросс-фиттинг), гетерогенность по категориям (CATE),
причинная кривая доза-отклик, конформные интервалы, обратная задача и плацебо-проверка.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from config import (COL_SKU, COL_CAT, RANDOM_STATE, DML_FOLDS, GRID, TEST_MONTHS)
from src.features import add_features, SEASON_COLS, LAG_COLS


def build_design(panel_feat: pd.DataFrame, emb_cols=None, t_col="own_disc"):
    emb_cols = emb_cols or [c for c in panel_feat.columns if c.startswith("emb_")]
    cat_dum = pd.get_dummies(panel_feat[COL_CAT].astype(str), prefix="cat")
    W = pd.concat([panel_feat[SEASON_COLS + LAG_COLS + emb_cols].reset_index(drop=True),
                   cat_dum.reset_index(drop=True)], axis=1).astype(float)
    Y = panel_feat["y"].astype(float).values
    T = panel_feat[t_col].astype(float).values
    sku = panel_feat[COL_SKU].values
    return Y, T, W, sku


# Double ML (PLR)
def dml_plr(Y, T, Wdf, cluster, nfold=DML_FOLDS, seed=RANDOM_STATE):
    Wv = Wdf.values if hasattr(Wdf, "values") else np.asarray(Wdf)
    n = len(Y); kf = KFold(nfold, shuffle=True, random_state=seed)
    Yr = np.zeros(n); Tr = np.zeros(n)
    for trn, tst in kf.split(Wv):
        Yr[tst] = Y[tst] - HistGradientBoostingRegressor(random_state=seed).fit(Wv[trn], Y[trn]).predict(Wv[tst])
        Tr[tst] = T[tst] - HistGradientBoostingRegressor(random_state=seed).fit(Wv[trn], T[trn]).predict(Wv[tst])
    try:                                                # точные кластерные SE
        import statsmodels.api as sm
        res = sm.OLS(Yr, sm.add_constant(Tr)).fit(cov_type="cluster", cov_kwds={"groups": cluster})
        ci = res.conf_int()[1]
        return res.params[1], res.bse[1], res.pvalues[1], ci[0], ci[1]
    except Exception:                                   # запасной путь: наклон + бутстрэп
        theta = np.polyfit(Tr, Yr, 1)[0]
        rng = np.random.RandomState(seed); uniq = np.unique(cluster)
        idx = {c: np.where(cluster == c)[0] for c in uniq}
        boot = []
        for _ in range(200):
            s = rng.choice(uniq, len(uniq), replace=True)
            pos = np.concatenate([idx[c] for c in s])
            boot.append(np.polyfit(Tr[pos], Yr[pos], 1)[0])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return theta, np.std(boot), np.nan, lo, hi


def cate_by_category(Y, T, W, sku, cats, min_rows=400, min_treated=60):
    rows = []
    for c in pd.unique(cats):
        m = (cats == c)
        if m.sum() < min_rows or (T[m] > 0).sum() < min_treated:
            continue
        th, _, _, lo, hi = dml_plr(Y[m], T[m], W.loc[m], sku[m])
        rows.append({"категория": c, "theta": th, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).sort_values("theta", ascending=False).reset_index(drop=True)


# доза-отклик
def _hinge(t):
    t = np.asarray(t, float)
    return np.column_stack([t, np.maximum(t - 10, 0), np.maximum(t - 20, 0)])


def dose_response(Y, T, W, sku, theta, lo, hi, grid=None, nboot=200, seed=RANDOM_STATE):

    grid = np.array(grid if grid is not None else GRID + [35, 40], float)
    lin    = (np.exp(theta * grid) - 1) * 100
    lin_lo = (np.exp(lo * grid) - 1) * 100
    lin_hi = (np.exp(hi * grid) - 1) * 100
    Wv = W.values; n = len(Y); B = _hinge(T)
    kf = KFold(5, shuffle=True, random_state=seed)
    Yr = np.zeros(n); Br = np.zeros_like(B)
    for trn, tst in kf.split(Wv):
        Yr[tst] = Y[tst] - HistGradientBoostingRegressor(random_state=seed).fit(Wv[trn], Y[trn]).predict(Wv[tst])
        for j in range(B.shape[1]):
            Br[tst, j] = B[tst, j] - HistGradientBoostingRegressor(random_state=seed).fit(Wv[trn], B[trn, j]).predict(Wv[tst])
    beta = np.linalg.lstsq(Br, Yr, rcond=None)[0]
    Bg = _hinge(grid)
    nl = (np.exp(Bg @ beta) - 1) * 100
    rng = np.random.RandomState(seed); uniq = np.unique(sku)
    idx = {s: np.where(sku == s)[0] for s in uniq}; boot = []
    for _ in range(nboot):
        s = rng.choice(uniq, len(uniq), replace=True)
        pos = np.concatenate([idx[x] for x in s])
        b = np.linalg.lstsq(Br[pos], Yr[pos], rcond=None)[0]
        boot.append((np.exp(Bg @ b) - 1) * 100)
    boot = np.array(boot)
    return pd.DataFrame({"grid": grid, "lin": lin, "lin_lo": lin_lo, "lin_hi": lin_hi,
                         "nl": nl, "nl_lo": np.percentile(boot, 2.5, axis=0),
                         "nl_hi": np.percentile(boot, 97.5, axis=0)}), boot


def inverse_discount(curve_df, boot, targets=(10, 20, 30, 50)):
    grid = curve_df["grid"].values; point = curve_df["nl"].values
    out = []
    for t in targets:
        j = np.where(point >= t)[0]
        d = grid[j[0]] if len(j) else None
        ds = [grid[np.where(c >= t)[0][0]] if (c >= t).any() else np.nan for c in boot]
        ds = np.array(ds)
        band = (np.nanpercentile(ds, 10), np.nanpercentile(ds, 90)) if not np.all(np.isnan(ds)) else (np.nan, np.nan)
        out.append({"цель_%": t, "скидка_%": d, "диапазон_low": band[0], "диапазон_high": band[1]})
    return pd.DataFrame(out)


# неопределённость
def conformal_coverage(panel_feat, W, t_col="own_disc", test_months=TEST_MONTHS, seed=RANDOM_STATE):
    """Split-conformal 90%-интервал: фактическое покрытие и полуширина на отложенных месяцах."""
    months = np.sort(panel_feat["month"].unique())
    cal_cut, test_cut = months[-2 * test_months], months[-test_months]
    Y = panel_feat["y"].values
    X = pd.concat([W.reset_index(drop=True), panel_feat[[t_col]].reset_index(drop=True)], axis=1).values
    tr = (panel_feat["month"] < cal_cut).values
    cal = ((panel_feat["month"] >= cal_cut) & (panel_feat["month"] < test_cut)).values
    te = (panel_feat["month"] >= test_cut).values
    base = HistGradientBoostingRegressor(random_state=seed).fit(X[tr], Y[tr])
    q = np.quantile(np.abs(Y[cal] - base.predict(X[cal])), 0.9)
    pred = base.predict(X[te])
    cover = np.mean((Y[te] >= pred - q) & (Y[te] <= pred + q))
    return float(cover), float(q)


# валидация
def placebo_test(Y, T, W, sku, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    th, _, _, lo, hi = dml_plr(Y, rng.permutation(T), W, sku)
    return th, lo, hi
