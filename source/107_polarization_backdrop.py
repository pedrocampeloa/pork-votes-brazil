# -*- coding: utf-8 -*-
"""
107_polarization_backdrop.py — simple, replicable polarization metric
=====================================================================
Builds a simple and self-contained descriptive measure of legislative
polarization for the sample window (2015-2022), suitable to serve as
institutional backdrop in the Institutional Setting section. Two
measures, both defined only from roll-call outcomes and the
government-vs-opposition classification already used in the paper:

1. Mean government-opposition vote-share distance:
     For each substantive vote, compute
       gap_v = | share_gov_sim - share_opp_sim |
     Average within year. 0 = both sides voted identically; 1 = perfect
     confrontation.

2. Fraction of "polarized" votes:
     For each substantive vote, mark it polarized if >=70% of one side
     votes "Sim" and >=70% of the other votes "Não" (or vice versa).
     Report the annual fraction.

Both metrics are trivially replicable from the raw panel + the
government-orientation labels already documented in Section 3, and
neither depends on any external paper or methodology.

Outputs
-------
docs/figs/fig_polarization_backdrop.pdf
results/polarization_backdrop.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import _utils as U

REPO = Path(__file__).resolve().parent.parent
FIG_OUT = REPO / "docs" / "figs"
CSV_OUT = REPO / "results"
FIG_OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)


def build_metric(df):
    """Return long DataFrame of (idVotacao, date, gap, polarized_70)."""
    sub = df[df["voto"].isin(["Sim", "Não"])].copy()
    sub["voto_sim"] = (sub["voto"] == "Sim").astype(int)

    grp = sub.groupby(["idVotacao", "coalizao_status"])["voto_sim"].agg(
        ["mean", "size"]).reset_index()
    grp.columns = ["idVotacao", "coalizao_status", "mean_sim", "n"]

    piv = grp.pivot(index="idVotacao", columns="coalizao_status",
                     values="mean_sim").reset_index()
    piv.columns.name = None
    sizes = grp.pivot(index="idVotacao", columns="coalizao_status",
                       values="n").reset_index()
    sizes.columns.name = None
    sizes = sizes.rename(columns={"coalizao": "n_coalizao",
                                    "oposicao": "n_oposicao"})
    piv = piv.merge(sizes[["idVotacao", "n_coalizao", "n_oposicao"]],
                     on="idVotacao")

    dates = df[["idVotacao", "data"]].drop_duplicates("idVotacao")
    piv = piv.merge(dates, on="idVotacao", how="left")
    piv["date"] = pd.to_datetime(piv["data"])
    piv["year"] = piv["date"].dt.year

    piv = piv[(piv["n_coalizao"] >= 20) & (piv["n_oposicao"] >= 20)].copy()
    piv["gap"] = (piv["coalizao"] - piv["oposicao"]).abs()
    piv["polarized_70"] = (
        ((piv["coalizao"] >= 0.7) & (piv["oposicao"] <= 0.3)) |
        ((piv["coalizao"] <= 0.3) & (piv["oposicao"] >= 0.7))
    ).astype(int)

    return piv


def make_figure(metric_df):
    yr = metric_df.groupby("year").agg(
        n_votes=("gap", "size"),
        mean_gap=("gap", "mean"),
        pol70=("polarized_70", "mean"),
    ).reset_index()

    # Presidential sub-periods for shading
    subperiods = [
        (2015.0, 2016.66, "Rousseff / Temer", "#e8f0f6"),
        (2016.66, 2018.99, "Temer",           "#dde8f2"),
        (2019.0, 2022.99, "Bolsonaro",        "#f5e3dc"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    for x0, x1, label, color in subperiods:
        ax.axvspan(x0, x1, color=color, alpha=0.6, zorder=0)
        ax.text((x0 + x1) / 2, 0.72, label, ha="center", va="top",
                fontsize=8, color="gray", zorder=1)

    ax.plot(yr["year"], yr["mean_gap"], "o-",
            color="#1f4e79", linewidth=1.8, markersize=6, zorder=3,
            label="Mean vote-share gap (gov vs opp)")
    ax.plot(yr["year"], yr["pol70"], "s--",
            color="#8b0000", linewidth=1.4, markersize=5, zorder=3,
            label=r"Share of polarized votes ($\geq$70/30 split each side)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Polarization measure")
    ax.set_ylim(0, 0.75)
    ax.set_xticks(yr["year"])
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out = FIG_OUT / "fig_polarization_backdrop.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}", flush=True)

    return yr


def main():
    print("Loading modeling panel...", flush=True)
    df = U.load_modeling_panel()
    metric = build_metric(df)
    print(f"n votes with valid gov/opp classification: {len(metric):,}",
          flush=True)
    yr = make_figure(metric)
    yr.to_csv(CSV_OUT / "polarization_backdrop.csv", index=False)
    print(yr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {CSV_OUT / 'polarization_backdrop.csv'}", flush=True)


if __name__ == "__main__":
    main()
