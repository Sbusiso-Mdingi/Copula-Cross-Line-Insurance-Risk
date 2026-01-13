"""
Exploratory Data Analysis (EDA) for synthetic insurance losses.

Purpose:
- Validate distributional assumptions
- Diagnose tail behaviour
- Justify EVT usage for property losses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from src.config import OUTPUT_PATHS

# ======================================================
# Load Data
# ======================================================

DATA_PATH = f"{OUTPUT_PATHS['simulations']}synthetic_annual_losses.csv"

df = pd.read_csv(DATA_PATH)

life = df["life_losses"]
property_ = df["property_losses"]

# ======================================================
# Utility Functions
# ======================================================

def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATHS['figures']}{name}", dpi=300)
    plt.close()


# ======================================================
# 1. Basic Distribution Plots
# ======================================================

def plot_histograms():
    plt.figure()
    plt.hist(life, bins=20, density=True)
    plt.title("Life Insurance Losses – Histogram")
    save_fig("life_histogram.png")

    plt.figure()
    plt.hist(property_, bins=20, density=True)
    plt.title("Property Insurance Losses – Histogram")
    save_fig("property_histogram.png")


# ======================================================
# 2. Log-Scale Diagnostics (Skewness)
# ======================================================

def plot_log_losses():
    plt.figure()
    plt.hist(np.log(life), bins=20, density=True)
    plt.title("Log Life Losses")
    save_fig("life_log_histogram.png")

    plt.figure()
    plt.hist(np.log(property_), bins=20, density=True)
    plt.title("Log Property Losses")
    save_fig("property_log_histogram.png")


# ======================================================
# 3. QQ Plots (Tail Diagnostics)
# ======================================================

def qq_plots():
    # --- Life: lognormal ---
    shape, loc, scale = stats.lognorm.fit(life, floc=0)

    plt.figure()
    stats.probplot(
        life,
        dist=stats.lognorm(s=shape, loc=loc, scale=scale),
        plot=plt
    )
    plt.title("Life Losses – Lognormal QQ Plot")
    save_fig("life_qq_lognormal.png")

    # --- Property: lognormal (diagnostic only) ---
    shape_p, loc_p, scale_p = stats.lognorm.fit(property_, floc=0)

    plt.figure()
    stats.probplot(
        property_,
        dist=stats.lognorm(s=shape_p, loc=loc_p, scale=scale_p),
        plot=plt
    )
    plt.title("Property Losses – Lognormal QQ Plot")
    save_fig("property_qq_lognormal.png")

# ======================================================
# 4. Mean Excess Plot (EVT Justification)
# ======================================================

def mean_excess_plot(data, title, filename):
    thresholds = np.linspace(
        np.percentile(data, 70),
        np.percentile(data, 98),
        30
    )

    mean_excess = [
        np.mean(data[data > u] - u) for u in thresholds
    ]

    plt.figure()
    plt.plot(thresholds, mean_excess, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Excess")
    plt.title(title)
    save_fig(filename)


# ======================================================
# 5. Correlation Check (Pre-Copula)
# ======================================================

def raw_correlation():
    corr = np.corrcoef(life, property_)[0, 1]
    print(f"Raw Pearson correlation (life vs property): {corr:.3f}")


# ======================================================
# Main Execution
# ======================================================

if __name__ == "__main__":

    plot_histograms()
    plot_log_losses()
    qq_plots()
    mean_excess_plot(
        property_,
        "Property Losses – Mean Excess Plot",
        "property_mean_excess.png"
    )
    raw_correlation()

    print("EDA completed. Figures saved.")
