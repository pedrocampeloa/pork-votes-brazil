# -*- coding: utf-8 -*-
"""
106_enriched_nuisance.py — Enrich the nuisance function
========================================================
Adds tipo × tema interactions to the control set so the ElasticNet
nuisance regressions absorb more of the vote-level heterogeneity.
The rationale is that the SE two-way explodes because V_vote is very
high in the Bolsonaro sample; if the nuisance function captures more
of that variation, the residual score has less clustered structure
and the CGM SE contracts.

Runs 1-way first (cheap) to check that the point estimate is stable
with the enriched controls. If stable, produces 2-way SE by post-hoc
CGM on the same score.

Uses n_folds=3, n_reps=1 for speed.

Outputs
-------
results/twoway_clustering/enriched_nuisance.csv
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


def build_enriched_panel(df):
    """Add tipo × tema interactions to the panel."""
    df = df.copy()
    tipo_cols = [c for c in df.columns if c.startswith('d_tipoVotacao_')]
    tema_cols = [c for c in df.columns if c.startswith('d_tema_')]

    added = []
    for t in tipo_cols:
        if df[t].std() == 0: continue
        for th in tema_cols:
            if df[th].std() == 0: continue
            interaction = df[t] * df[th]
            # Skip if too few non-zero (sparse interaction)
            if (interaction > 0).sum() < 500:
                continue
            new_col = f"int_{t}_x_{th}"
            df[new_col] = interaction.astype(np.float32)
            added.append(new_col)
    return df, added


def cluster_sums_of_squares(psi, cluster1, cluster2):
    V_d = float(pd.Series(psi).groupby(cluster1).sum().pow(2).sum())
    V_v = float(pd.Series(psi).groupby(cluster2).sum().pow(2).sum())
    inter = pd.MultiIndex.from_arrays([cluster1, cluster2])
    V_dv = float(pd.Series(psi).groupby(inter).sum().pow(2).sum())
    return V_d, V_v, V_dv


def fit_pliv_enriched(df, target, n_folds=3, n_reps=1):
    from doubleml import DoubleMLClusterData, DoubleMLPLIV
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    ivs = list(C.IV_SETS["backlog"])
    ivs = [z for z in ivs if z in df.columns and df[z].std() > 0]

    # Use ALL controls: current controls + interaction terms
    controls = UV.get_clean_full_controls(df)
    interaction_cols = [c for c in df.columns if c.startswith('int_')]
    controls = list(dict.fromkeys(controls + interaction_cols))
    controls = [c for c in controls
                if c not in (target, C.TREATMENT, "idDeputado", "idVotacao")
                and c not in ivs]
    controls = [c for c in controls if c in df.columns
                and df[c].notna().mean() > 0.5
                and df[c].nunique() > 1]

    print(f"  n controls after enrichment: {len(controls)}", flush=True)

    cols = [target, C.TREATMENT] + controls + ivs + ["idDeputado", "idVotacao"]
    seen, cu = set(), []
    for c in cols:
        if c not in seen: cu.append(c); seen.add(c)
    work = df[cu].dropna().copy()

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
                        work[["idDeputado"]]], axis=1)

    data = DoubleMLClusterData(
        df_dml, y_col=target, d_cols=C.TREATMENT,
        cluster_cols="idDeputado",
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
    pliv.fit()

    return pliv, work, sc_t, controls


def run_leg(df_enriched, leg, n_folds=3, n_reps=1):
    from scipy.stats import norm
    print(f"\n=== LEG {leg} | enriched nuisance | n_folds={n_folds}, n_reps={n_reps} ===", flush=True)
    sub = df_enriched[df_enriched["idLegislatura"] == leg].copy()
    print(f"  n_obs = {len(sub):,}", flush=True)

    t0 = time.time()
    pliv, work, sc_t, controls = fit_pliv_enriched(sub, "alinhamento", n_folds, n_reps)
    dt = time.time() - t0
    print(f"  fit done in {dt:.1f}s", flush=True)

    psi = pliv.psi.reshape(-1)
    psi_deriv = pliv.psi_deriv.reshape(-1)
    J = -np.mean(psi_deriv)
    n = len(psi)
    coef = float(pliv.coef[0])
    std_t = float(sc_t.scale_[0])
    deputy_ids = work["idDeputado"].values
    vote_ids = work["idVotacao"].values

    V_d, V_v, V_dv = cluster_sums_of_squares(psi, deputy_ids, vote_ids)
    V_2w = V_d + V_v - V_dv

    se_1w = np.sqrt(V_d) / abs(J) / n
    se_2w = np.sqrt(V_2w) / abs(J) / n

    to_pp = lambda se: 100 * se / std_t
    pp = 100 * coef / std_t
    pp_se_1w = to_pp(se_1w)
    pp_se_2w = to_pp(se_2w)

    def stats(pp_se):
        z = pp / pp_se
        p = 2 * (1 - norm.cdf(abs(z)))
        s = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        return z, p, s

    z1, p1, s1 = stats(pp_se_1w)
    z2, p2, s2 = stats(pp_se_2w)

    print(f"  coef = {pp:+.4f} pp/R$M")
    print(f"  V_v / V_d = {V_v/V_d:.2f}  (was 1.82 in Leg 55, 14.21 in Leg 56 without enrichment)")
    print(f"  1-way SE={pp_se_1w:.4f}  CI=[{pp-1.96*pp_se_1w:+.3f},{pp+1.96*pp_se_1w:+.3f}]  p={p1:.4f} {s1}")
    print(f"  2-way SE={pp_se_2w:.4f}  CI=[{pp-1.96*pp_se_2w:+.3f},{pp+1.96*pp_se_2w:+.3f}]  p={p2:.4f} {s2}")
    print(f"  ratio 2w/1w = {pp_se_2w/pp_se_1w:.3f}")

    return {
        "leg": leg, "n_obs": n,
        "coef_sd": coef, "pp_per_unit": pp,
        "n_controls": len(controls),
        "V_d": V_d, "V_v": V_v, "V_dv": V_dv,
        "V_v_over_V_d": V_v / V_d,
        "pp_se_1way": pp_se_1w, "pp_se_2way": pp_se_2w,
        "se_ratio": pp_se_2w / pp_se_1w,
        "ci_1w_lo": pp - 1.96 * pp_se_1w, "ci_1w_hi": pp + 1.96 * pp_se_1w,
        "ci_2w_lo": pp - 1.96 * pp_se_2w, "ci_2w_hi": pp + 1.96 * pp_se_2w,
        "pval_1w": p1, "pval_2w": p2,
        "stars_1w": s1, "stars_2w": s2,
        "fit_seconds": dt,
    }


def main():
    print("Loading modeling panel...", flush=True)
    df = U.load_modeling_panel()
    print(f"panel: n={len(df):,}", flush=True)

    print("\nBuilding enriched panel with tipo x tema interactions...", flush=True)
    df_enr, added = build_enriched_panel(df)
    print(f"  added {len(added)} interaction terms", flush=True)

    rows = []
    for leg in (55, 56):
        r = run_leg(df_enr, leg, n_folds=3, n_reps=1)
        rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "enriched_nuisance.csv", index=False)

    print(f"\nwrote {OUT / 'enriched_nuisance.csv'}", flush=True)


if __name__ == "__main__":
    main()
