""""
CVT Temperature Over Time — Logarithmic Regression
-------------------------------------------------------
Same pipeline as before (Analog1 volts -> Temperature via the
calibration line), but instead of fitting a straight line to
Temperature vs Time, this fits a LOGARITHMIC curve:

    Temperature(t) = a * ln(t) + b

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
LOCAL_CALIBRATION_FILE = BASE_DIR / "calibration_data.txt"
CALIBRATION_FILE = (
    LOCAL_CALIBRATION_FILE
    if LOCAL_CALIBRATION_FILE.exists()
    else BASE_DIR.parent / "yellow_4th" / "calibration_data.txt"
)
TRIAL_FILES = [BASE_DIR / "yellowf_4th.csv"]
OUTPUT_DIR = BASE_DIR / "output_plots"
ANALOG1_COL_INDEX = 3       # "Analog1" (volts)
ELAPSED_TIME_COL_INDEX = 2  # "ElapsedTime" (minutes)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# STEP 1 — Fit the calibration curve: Temperature = m * Volts + b
cal = pd.read_csv(CALIBRATION_FILE, sep="\t").dropna()
cal_fit = stats.linregress(cal["Volts"].to_numpy(), cal["Temperature (Fahrenheit)"].to_numpy())
slope, intercept = cal_fit.slope, cal_fit.intercept


def volts_to_temp(volts):
    return slope * volts + intercept


for trial_path in TRIAL_FILES:
    trial_name = trial_path.stem

    df = pd.read_csv(trial_path, skiprows=1, header=None)

    mask = pd.to_numeric(df[ANALOG1_COL_INDEX], errors="coerce").notna()
    volts = pd.to_numeric(df.loc[mask, ANALOG1_COL_INDEX], errors="coerce")
    time = pd.to_numeric(df.loc[mask, ELAPSED_TIME_COL_INDEX], errors="coerce")
    temps = volts_to_temp(volts)

    # Drop any rows where time <= 0, since ln(time) is undefined there
    valid = time > 0
    time, temps = time[valid], temps[valid]

    # ---- Logarithmic regression: Temp = a*ln(time) + b ----
    log_time = np.log(time)
    log_fit = stats.linregress(log_time, temps)
    a, b = log_fit.slope, log_fit.intercept

    # ---- Scatter plot with log regression curve ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(time, temps, alpha=0.5, s=8, color="darkorange", label="Temperature readings")

    x_line = np.linspace(time.min(), time.max(), 200)
    y_line = a * np.log(x_line) + b
    ax.plot(x_line, y_line, color="navy", linewidth=2,
            label=f"Log fit: T={a:.3f}*ln(t)+{b:.2f}\nR^2={log_fit.rvalue**2:.3f}")

    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylabel("Temperature (F)")
    ax.set_title(f"{trial_name}: Temperature Over Time (Logarithmic Fit, n={len(temps)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{trial_name}_temp_vs_time_log.png", dpi=150)
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
          f"Log fit: T={a:.3f}*ln(t)+{b:.2f}  (R^2={log_fit.rvalue**2:.3f})")

print(f"\nAll plots saved to {OUTPUT_DIR}/")
