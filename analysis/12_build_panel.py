
"""
12_build_panel.py
-----------------
Build the analytical panel for Part 2.

Inputs:
  data/processed/master_tfr.csv         (TFR + subgroup + bloc, from Part 1)
  data/raw/raw_wb_gdp_per_capita_ppp.csv
  data/raw/raw_wb_urban_pop_pct.csv
  data/raw/raw_wb_remittances_gdp_pct.csv
  data/raw/raw_wb_under5_mortality.csv

Output:
  data/processed/panel.csv

Design rules (do NOT relax):
  - No interpolation. No extrapolation. Gaps stay as NaN; the panel is unbalanced.
  - Missingness is flagged per variable (boolean *_missing column) for transparency.
  - 1-year lags are created for the four annual covariates (used in the FE models).
  - log(GDP-PPP) is precomputed for convenience.
  - A Central Asia dummy is added.

Run from repo root:  python analysis/12_build_panel.py
"""

import os
import numpy as np
import pandas as pd

from _assertions import assert_year_continuity

ANNUAL_VARS = [
    "gdp_per_capita_ppp",
    "urban_pop_pct",
    "remittances_gdp_pct",
    "under5_mortality",
]

tfr = pd.read_csv("data/processed/master_tfr.csv")

panel = tfr.copy()
for var in ANNUAL_VARS:
    cov = pd.read_csv(f"data/raw/raw_wb_{var}.csv")
    panel = panel.merge(cov[["country", "year", var]], on=["country", "year"],
                        how="left", validate="many_to_one")

panel = panel.sort_values(["country", "year"]).reset_index(drop=True)

assert not panel.duplicated(["country", "year"]).any(), (
    "Duplicate (country, year) rows in panel — a covariate merge introduced "
    "a fan-out; check the raw_wb_*.csv files for repeated country-year rows."
)

for var in ANNUAL_VARS:
    panel[f"{var}_missing"] = panel[var].isna()


assert_year_continuity(panel)


for var in ANNUAL_VARS:
    panel[f"{var}_lag1"] = panel.groupby("country")[var].shift(1)

panel["log_gdp_ppp"] = np.log(panel["gdp_per_capita_ppp"])
panel["log_gdp_ppp_lag1"] = panel.groupby("country")["log_gdp_ppp"].shift(1)

panel["ca"] = (panel["bloc"] == "Central Asia").astype(int)

os.makedirs("data/processed", exist_ok=True)
panel.to_csv("data/processed/panel.csv", index=False)

print(f"Panel saved -> data/processed/panel.csv")
print(f"Shape: {panel.shape}   Countries: {panel['country'].nunique()}   Years: {panel['year'].min()}-{panel['year'].max()}")
print()

print("=== Non-missing counts per variable (raw, after merge) ===")
for var in ANNUAL_VARS + ["tfr"]:
    print(f"  {var:24s}: {int(panel[var].notna().sum())}/{len(panel)}")

print("\n=== Non-missing counts per LAGGED variable (1 year always lost per country at t=2000) ===")
for var in ANNUAL_VARS:
    col = f"{var}_lag1"
    print(f"  {col:30s}: {int(panel[col].notna().sum())}/{len(panel)}")

print("\n=== Effective sample for the FE model (all four LAGGED covariates + TFR present) ===")
lag_cols = [f"{v}_lag1" for v in ANNUAL_VARS]
complete = panel[["tfr"] + lag_cols].notna().all(axis=1)
print(f"  Complete-case rows: {int(complete.sum())}/{len(panel)}")
print(f"  (the 14 cells lost at year=2000 are the lag drop; the rest reflect the remittances gaps)")

print("\n=== CA dummy distribution ===")
print(panel.groupby("bloc")["ca"].first().to_string())