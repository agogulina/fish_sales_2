"""Слой 3. Оптимизатор промо: экономика, оптимизатор, off-policy при равном бюджете."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from config import COL_CAT, COL_SKU, OUT_DIR
from src.data import build_panel
from src.features import add_features
from src.layer2_causal import build_design, dml_plr
from src.layer3_optimizer import (attach_prices, baseline_table, make_econ,
                                  candidates, optimize, off_policy)

if __name__ == "__main__":
    panel = attach_prices(build_panel())
    pf = add_features(panel)

    # причинный эффект по категориям (как в Слое 2)
    Y, T, W, sku = build_design(pf)
    theta_glob = dml_plr(Y, T, W, sku)[0]
    theta_cat = {}
    for c in pf[COL_CAT].unique():
        m = (pf[COL_CAT].values == c)
        if m.sum() < 400 or (T[m] > 0).sum() < 60:
            continue
        th, _, _, lo, hi = dml_plr(Y[m], T[m], W.loc[m], sku[m])
        theta_cat[c] = (th, lo, hi)
    print("theta по категориям:", len(theta_cat), "| глоб:", round(theta_glob, 4))

    base = baseline_table(panel)
    econ = make_econ(theta_cat, theta_glob)
    cand = candidates(base, econ)

    hist_spend, hist_contrib = off_policy(panel, base, econ)
    plan, method = optimize(cand, budget=hist_spend)   # равный бюджет с историей

    from config import COST_SHARE
    if len(plan) == 0:
        margin = round((1 - COST_SHARE) * 100)
        print(f"\nПри COST_SHARE={COST_SHARE} (маржа {margin}%) ни одна скидка не выгодна:")
        print("  ожидаемый вклад от любого промо отрицателен -> оптимум: НЕ давать скидок.")
        print(f"  историческая политика при этом даёт вклад: {hist_contrib:,.0f} (часть скидок убыточна).")
        print("  Уменьшите COST_SHARE в config.py, чтобы получить ненулевой план.")
    else:
        print(f"\nМетод: {method} | SKU в промо: {len(plan)} | вклад плана: {plan['net'].sum():,.0f}")
        print(f"История (равный бюджет): {hist_contrib:,.0f}")
        g = plan['net'].sum() - hist_contrib
        print(f"Прирост от оптимизации: {g:+,.0f} ({g/max(hist_contrib,1)*100:+.1f}%)")

        out = os.path.join(OUT_DIR, "promo_plan_layer3.xlsx")
        plan.merge(base[[COL_SKU, "cat"]].rename(columns={COL_SKU: "sku"}), on="sku", how="left") \
            .sort_values("net", ascending=False).round(0).to_excel(out, index=False)
        print("План сохранён:", out)
