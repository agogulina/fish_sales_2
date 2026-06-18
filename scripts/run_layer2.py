
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import COL_CAT
from src.data import build_panel, load_catalog
from src.features import add_features, e5_embeddings
from src.layer2_causal import (build_design, dml_plr, cate_by_category,
                               dose_response, inverse_discount, conformal_coverage, placebo_test)

if __name__ == "__main__":
    panel = build_panel()
    pf = add_features(panel)
    emb = e5_embeddings(load_catalog())          
    if emb is not None:
        pf = pf.merge(emb, on="Головное СКЮ Артикул", how="left")
        for c in [c for c in emb.columns if c.startswith("emb_")]:
            pf[c] = pf[c].fillna(0.0)

    Y, T, W, sku = build_design(pf)
    print("Наблюдений:", len(Y), "| доля со скидкой:", round((T > 0).mean(), 3))

    theta, se, p, lo, hi = dml_plr(Y, T, W, sku)
    print(f"\n[DML] эффект скидки theta={theta:+.4f}/п.п.  SE={se:.4f}  p={p}  CI=[{lo:+.4f},{hi:+.4f}]")
    print(f"      +10 п.п. -> {(np.exp(10*theta)-1)*100:+.1f}% продаж")

    cate = cate_by_category(Y, T, W, sku, pf[COL_CAT].values)
    print("\n[CATE] эффект по категориям (топ):"); print(cate.head(8).round(4).to_string(index=False))

    curve, boot = dose_response(Y, T, W, sku, theta, lo, hi)
    print("\n[Доза-отклик] линейно / нелинейно:")
    for _, r in curve.iterrows():
        print(f"  {int(r.grid):2d}%: {r.lin:+.0f}% / {r.nl:+.0f}%")

    inv = inverse_discount(curve, boot)
    print("\n[Обратная задача]"); print(inv.round(1).to_string(index=False))

    cover, q = conformal_coverage(pf, W)
    print(f"\n[Конформные интервалы] покрытие {cover:.1%} (цель 90%), полуширина {q:.3f}")

    th_p, lo_p, hi_p = placebo_test(Y, T, W, sku)
    print(f"\n[Плацебо] theta={th_p:+.4f} CI=[{lo_p:+.4f},{hi_p:+.4f}] -> ожидаемо около нуля")
