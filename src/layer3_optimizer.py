"""
Слой 3 — оптимизатор промо.
"""
import numpy as np
import pandas as pd

from config import (COL_SKU, COL_CLIENT, COL_CAT, GRID, COST_SHARE,
                    SLOTS, KMAX, ROBUST, PRICE_TABLE_XLSX)

def attach_prices(panel: pd.DataFrame, price_xlsx=PRICE_TABLE_XLSX):
    pr = pd.read_excel(price_xlsx, sheet_name="Цены по SKU")
    pr["Артикул"] = pr["Артикул"].astype(str)
    P = pr.set_index("Артикул")
    out = panel.copy()
    out["price_unit"] = out[COL_SKU].map(P["оценка_цена_входа_шт"])
    out["group"] = out[COL_SKU].map(P["Группа (Слой 1)"]).fillna("—")
    return out


def baseline_table(panel: pd.DataFrame):
    base = (panel[panel.own_disc == 0].groupby(COL_SKU)
            .agg(base_qty=("qty", "median"), cat=(COL_CAT, "first"),
                 group=("group", "first"), price_unit=("price_unit", "first")).reset_index())
    return base[base["price_unit"].notna() & (base["base_qty"] > 0)].copy()


def make_econ(theta_cat: dict, theta_glob: float, cost_share=COST_SHARE):
    def uplift(cat, d, which="point"):
        t = theta_cat.get(cat, (theta_glob, theta_glob, theta_glob))
        th = t[0] if which == "point" else t[1]
        return np.exp(th * d) - 1

    def econ(bq, pu, cat, d, which="point"):
        up = uplift(cat, d, which); nq = bq * (1 + up); cost = cost_share * pu
        contrib = nq * (pu * (1 - d / 100.0) - cost) - bq * (pu - cost)
        rev     = nq * pu * (1 - d / 100.0) - bq * pu
        spend   = nq * pu * (d / 100.0)
        return contrib, rev, spend
    return econ


def candidates(base: pd.DataFrame, econ, grid=GRID):
    rows = []
    for _, r in base.iterrows():
        for d in grid:
            c, rev, sp = econ(r["base_qty"], r["price_unit"], r["cat"], d)
            c_lo, _, _ = econ(r["base_qty"], r["price_unit"], r["cat"], d, "lo")
            rows.append(dict(sku=r[COL_SKU], cat=r["cat"], group=r["group"], d=d,
                             net=c, net_robust=c_lo, rev=rev, spend=sp))
    return pd.DataFrame(rows)

def optimize(cand: pd.DataFrame, budget, slots=SLOTS, kmax=KMAX, robust=ROBUST):
    obj = "net_robust" if robust else "net"
    pos = cand[(cand.d > 0) & (cand[obj] > 0)].reset_index(drop=True)
    try:
        import pulp
        prob = pulp.LpProblem("promo", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in pos.index}
        prob += pulp.lpSum(pos.loc[i, obj] * x[i] for i in pos.index)
        for s in pos.sku.unique():
            prob += pulp.lpSum(x[i] for i in pos.index[pos.sku == s]) <= 1
        prob += pulp.lpSum(pos.loc[i, "spend"] * x[i] for i in pos.index) <= budget
        prob += pulp.lpSum(x[i] for i in pos.index) <= slots
        for g in pos.group.unique():
            prob += pulp.lpSum(x[i] for i in pos.index[pos.group == g]) <= kmax
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        sel = [i for i in pos.index if x[i].value() == 1]
        return pos.loc[sel], "PuLP"
    except Exception as e:                              # noqa
        print("PuLP недоступен -> жадный:", repr(e))
        best = pos.sort_values(obj, ascending=False).groupby("sku").head(1).sort_values(obj, ascending=False)
        chosen, spent, gc, used = [], 0, {}, set()
        for _, r in best.iterrows():
            if r.sku in used or len(chosen) >= slots or spent + r.spend > budget or gc.get(r.group, 0) >= kmax:
                continue
            chosen.append(r); used.add(r.sku); spent += r.spend; gc[r.group] = gc.get(r.group, 0) + 1
        return pd.DataFrame(chosen), "жадный"


def off_policy(panel: pd.DataFrame, base: pd.DataFrame, econ):
    hist = (panel[panel.own_disc > 0].groupby(COL_SKU)
            .agg(d=("own_disc", "mean"), cat=(COL_CAT, "first")).reset_index()
            .merge(base[[COL_SKU, "base_qty", "price_unit"]], on=COL_SKU, how="inner"))
    he = hist.apply(lambda r: econ(r["base_qty"], r["price_unit"], r["cat"], r["d"]),
                    axis=1, result_type="expand")
    hist["net"], hist["spend"] = he[0], he[2]
    return float(hist["spend"].sum()), float(hist["net"].clip(lower=0).sum())
