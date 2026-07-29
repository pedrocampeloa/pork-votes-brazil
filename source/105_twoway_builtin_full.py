# -*- coding: utf-8 -*-
"""
105_twoway_builtin_full.py — Built-in two-way with n_folds=3, n_reps=3
======================================================================
Full canonical spec: cluster-aware cross-fitting on both cluster
dimensions (deputy and vote), with n_folds=3, n_reps=3, matching the
canonical spec used for the 1-way results throughout the paper.

Expected runtime is very high (n_folds^2 * n_reps = 27 folds per
nuisance function; ~22h total wall clock across both legislatures).
Runs Leg 55 first (~4h), then Leg 56 (~18h).

Outputs
-------
results/twoway_clustering/twoway_builtin_full.csv
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

# Reuse the run_leg from script 104
import importlib.util
spec = importlib.util.spec_from_file_location("s104", HERE / "104_twoway_builtin.py")
s104 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s104)


def main():
    print("Loading modeling panel...", flush=True)
    df = U.load_modeling_panel()
    print(f"panel: n={len(df):,}", flush=True)

    rows = []
    # Leg 55 (smaller)
    print("\n[1/2] Leg 55, canonical spec (n_folds=3, n_reps=3)", flush=True)
    t0 = time.time()
    r55 = s104.run_leg(df, 55, n_folds=3, n_reps=3)
    rows.append(r55)
    pd.DataFrame(rows).to_csv(OUT / "twoway_builtin_full.csv", index=False)
    print(f"  Leg 55 done, {time.time()-t0:.0f}s. Saved intermediate CSV.", flush=True)

    # Leg 56
    print("\n[2/2] Leg 56, canonical spec (n_folds=3, n_reps=3)", flush=True)
    t0 = time.time()
    r56 = s104.run_leg(df, 56, n_folds=3, n_reps=3)
    rows.append(r56)
    pd.DataFrame(rows).to_csv(OUT / "twoway_builtin_full.csv", index=False)
    print(f"  Leg 56 done, {time.time()-t0:.0f}s.", flush=True)
    print(f"\nwrote {OUT / 'twoway_builtin_full.csv'}", flush=True)


if __name__ == "__main__":
    main()
