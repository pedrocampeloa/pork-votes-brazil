# -*- coding: utf-8 -*-
"""
104_twoway_builtin.py — DoubleML built-in two-way clustering
=============================================================
Runs the PLIV-DML with cluster_cols=[idDeputado, idVotacao] passed
directly to DoubleMLClusterData, so the cross-fitting itself is
two-way cluster-aware (Chiang, Kato, Ma & Sasaki 2022, JBES),
not just the post-hoc variance.

To keep runtime tractable at our sample size (nfolds^2*nreps grows
fast), we use n_folds=2, n_reps=1. This still satisfies the DML
requirement n_folds >= 2 and gives 4 folds per nuisance function.

Comparison target:
  - Post-hoc CGM (from script 103): +1.887 [-0.48, +4.26] (Leg 55),
    -0.931 [-2.58, +0.72] (Leg 56).
  If the built-in is close, this validates the post-hoc equivalence
  argument of Chiang et al. under our data.

Outputs
-------
results/twoway_clustering/twoway_builtin.csv
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import _utils as U
import _config as C
import _utils_v2 as UV

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent.parent / "results" / "twoway_clustering"
OUT.mkdir(parents=True, exist_ok=True)


def run_leg(df, leg, n_folds=2, n_reps=1):
    from doubleml import DoubleMLClusterData, DoubleMLPLIV
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    print(f"\n=== LEG {leg} | built-in two-way | n_folds={n_folds}, n_reps={n_reps} ===", flush=True)
    sub = df[df["idLegislatura"] == leg].copy()
    target = "alinhamento"

    ivs = list(C.IV_SETS["backlog"])
    ivs = [z for z in ivs if z in sub.columns and sub[z].std() > 0]
    controls = UV.get_clean_full_controls(sub)
    controls = [c for c in controls
                if c not in (target, C.TREATMENT, "idDeputado", "idVotacao")
                and c not in ivs]
    controls = [c for c in controls if c in sub.columns
                and sub[c].notna().mean() > 0.5
                and sub[c].nunique() > 1]

    cols = [target, C.TREATMENT] + controls + ivs + ["idDeputado", "idVotacao"]
    seen, cu = set(), []
    for c in cols:
        if c not in seen: cu.append(c); seen.add(c)
    work = sub[cu].dropna().copy()

    cols_dem = [target, C.TREATMENT] + controls + ivs
    work = UV.within_transform(work, "idDeputado", cols_dem)

    sc_t = StandardScaler(); sc_x = StandardScaler(); sc_z = StandardScaler()
    T_s = pd.DataFrame(sc_t.fit_transform(work[[C.TREATMENT]]),
                        columns=[C.TREATMENT], index=work.index)
    X_s = pd.DataFrame(sc_x.fit_transform(work[controls]),
                        columns=controls, index=work.index)
    Z_s = pd.DataFrame(sc_z.fit_transform(work[ivs]),
                        columns=ivs, index=work.index)
    df_dml = pd.concat([work[[target]], T_s, X_s, Z_s,
                        work[["idDeputado", "idVotacao"]]], axis=1)

    # KEY DIFFERENCE FROM SCRIPT 101: pass BOTH cluster columns to
    # DoubleMLClusterData, so cross-fitting respects both dimensions.
    data = DoubleMLClusterData(
        df_dml, y_col=target, d_cols=C.TREATMENT,
        cluster_cols=["idDeputado", "idVotacao"],
        x_cols=list(controls), z_cols=list(ivs)
    )
    kw = dict(l1_ratio=[0.1, 0.5, 1.0], alphas=np.logspace(-3, 1, 10),
              cv=3, max_iter=2000, precompute=False)
    pliv = DoubleMLPLIV(
        data,
        ml_l=ElasticNetCV(**kw),
        ml_m=ElasticNetCV(**kw),
        ml_r=ElasticNetCV(**kw),
        n_folds=n_folds, n_rep=n_reps
    )
    t0 = time.time()
    pliv.fit()
    dt = time.time() - t0
    print(f"  fit done in {dt:.1f}s", flush=True)

    coef = float(pliv.coef[0])
    se = float(pliv.summary["std err"].iloc[0])
    pval = float(pliv.summary["P>|t|"].iloc[0])
    std_t = float(sc_t.scale_[0])
    n_obs = len(work)
    pp = 100 * coef / std_t
    pp_se = 100 * se / std_t
    ci_lo = pp - 1.96 * pp_se
    ci_hi = pp + 1.96 * pp_se
    stars = ("***" if pval < 0.01 else "**" if pval < 0.05
             else "*" if pval < 0.10 else "")

    print(f"  coef = {pp:+.4f} pp/R$M", flush=True)
    print(f"  SE (built-in two-way) = {pp_se:.4f}", flush=True)
    print(f"  95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]", flush=True)
    print(f"  p-value = {pval:.4f}  {stars}", flush=True)

    return {
        "leg": leg,
        "n_folds": n_folds, "n_reps": n_reps,
        "coef_sd": coef, "se_sd_builtin": se,
        "pp_per_unit": pp, "pp_se_builtin": pp_se,
        "ci95_lo_pp": ci_lo, "ci95_hi_pp": ci_hi,
        "pval": pval, "stars": stars,
        "n_obs": n_obs,
        "fit_seconds": dt,
    }


def main():
    print("Loading modeling panel...", flush=True)
    df = U.load_modeling_panel()
    print(f"panel: n={len(df):,}", flush=True)

    rows = []
    # Leg 55 first (smaller, faster). If it terminates in reasonable time,
    # continue to Leg 56.
    r55 = run_leg(df, 55, n_folds=2, n_reps=1)
    rows.append(r55)
    print(f"\n  Leg 55 done in {r55['fit_seconds']:.0f}s", flush=True)

    r56 = run_leg(df, 56, n_folds=2, n_reps=1)
    rows.append(r56)

    tab = pd.DataFrame(rows)
    csv_path = OUT / "twoway_builtin.csv"
    tab.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
