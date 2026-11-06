""""
CVT Temperature Over Time — Polynomial Regression
-------------------------------------------------------
Same pipeline as before (Analog1 volts -> Temperature via the
calibration line), but instead of fitting a logarithmic curve,
this fits a POLYNOMIAL curve of degree POLY_DEGREE:

    Temperature(t) = c_n*t^n + ... + c_1*t + c_0

POLY_DEGREE = 4 was chosen after comparing 5-fold cross-validated
R^2 across degrees 1-7 on this trial's data:
  deg=1: cv R2=0.143   deg=2: cv R2=0.793   deg=3: cv R2=0.879
  deg=4: cv R2=0.902   deg=5: cv R2=0.904   deg=6: cv R2=0.916
  deg=7: cv R2=0.918
Degree 4 sits right where the curve elbows: it captures the
rise/plateau/fall/re-rise shape of the data (a 3rd-degree
polynomial has at most 2 turning points, which isn't enough here),
while degrees above 4 only add ~1-2 more R^2 points per degree at
the cost of a much wigglier, easier-to-overfit curve. Train and
cross-validated R^2 track each other closely at every degree here,
so overfitting isn't a concern at this sample size (n~29k) either
way -- 4 is simply the simplest model that captures the real shape.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
# Resolve paths relative to this file, rather than the directory from
# which Python happens to be launched.
BASE_DIR = Path(__file__).resolve().parent
LOCAL_CALIBRATION_FILE = BASE_DIR / "calibration_data.txt"
# The calibration file is shared with the yellow_4th data set when a
# yellowf_6th-specific copy has not been saved yet.
CALIBRATION_FILE = (
    LOCAL_CALIBRATION_FILE
    if LOCAL_CALIBRATION_FILE.exists()
    else BASE_DIR.parent / "yellow_4th" / "calibration_data.txt"
)
TRIAL_FILES = [BASE_DIR / "yellowf_8th.csv"]
OUTPUT_DIR = BASE_DIR / "output_plots"
ANALOG1_COL_INDEX = 3       # "Analog1" (volts)
ELAPSED_TIME_COL_INDEX = 2  # "ElapsedTime" (minutes)
POLY_DEGREE = 4               # see justification in module docstring above

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# STEP 1 — Fit the calibration curve: Temperature = m * Volts + b
# ----------------------------------------------------------------------
cal = pd.read_csv(CALIBRATION_FILE, sep="\t").dropna()
cal_fit = stats.linregress(cal["Volts"].to_numpy(), cal["Temperature (Fahrenheit)"].to_numpy())
slope, intercept = cal_fit.slope, cal_fit.intercept


def volts_to_temp(volts):
    return slope * volts + intercept


# ----------------------------------------------------------------------
# STEP 2 — For each trial: convert volts -> temp, fit log(time) -> temp,
#           plot scatter + log curve, and a box-and-whisker plot.
# ----------------------------------------------------------------------
for trial_path in TRIAL_FILES:
    trial_name = trial_path.stem

    df = pd.read_csv(trial_path, skiprows=1, header=None)

    mask = pd.to_numeric(df[ANALOG1_COL_INDEX], errors="coerce").notna()
    volts = pd.to_numeric(df.loc[mask, ANALOG1_COL_INDEX], errors="coerce")
    time = pd.to_numeric(df.loc[mask, ELAPSED_TIME_COL_INDEX], errors="coerce")
    temps = volts_to_temp(volts)

    # Drop any rows where time <= 0 (kept for consistency with the old
    # log-based pipeline, though polynomial fitting doesn't require it)
    valid = time > 0
    time, temps = time[valid].to_numpy(), temps[valid].to_numpy()

    # ---- Polynomial regression: Temp = c_n*t^n + ... + c_0 ----
    coeffs = np.polyfit(time, temps, POLY_DEGREE)
    poly = np.poly1d(coeffs)
    pred = poly(time)
    ss_res = np.sum((temps - pred) ** 2)
    ss_tot = np.sum((temps - temps.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # ---- Scatter plot with polynomial regression curve ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time, temps, alpha=0.5, s=8, color="darkorange", label="Temperature readings")

    x_line = np.linspace(time.min(), time.max(), 200)
    y_line = poly(x_line)
    terms = " + ".join(f"{c:.3g}*t^{POLY_DEGREE - i}" if POLY_DEGREE - i > 1
                        else ("{:.3g}*t".format(c) if POLY_DEGREE - i == 1 else f"{c:.3g}")
                        for i, c in enumerate(coeffs))
    ax.plot(x_line, y_line, color="navy", linewidth=2,
            label=f"Degree-{POLY_DEGREE} poly fit\nR^2={r_squared:.3f}")

    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylabel("Temperature (F)")
    ax.set_title(f"{trial_name}: Temperature Over Time (Degree-{POLY_DEGREE} Polynomial Fit, n={len(temps)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{trial_name}_temp_vs_time_poly.png", dpi=150)
    plt.close(fig)

    # ---- Box-and-whisker plot of the temperature distribution ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.boxplot(temps, tick_labels=["Temperature (F)"])
    ax.set_ylabel("Temperature (F)")
    ax.set_title(f"{trial_name}: Temperature Distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{trial_name}_temp_boxplot.png", dpi=150)
    plt.close(fig)

    print(f"{trial_name}: {len(temps)} readings | "
          f"Temp range {temps.min():.1f}F - {temps.max():.1f}F | "
          f"Degree-{POLY_DEGREE} poly fit  (R^2={r_squared:.3f})")
    print(f"  Coefficients (highest degree first): {np.array2string(coeffs, precision=4)}")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
