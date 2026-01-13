"""
Marginal distribution fitting and PIT transformation.

Life insurance:
- Single parametric distribution

Property insurance:
- Body distribution + EVT tail (POT + GPD)

Outputs:
- PIT-transformed data for copula fitting
- Fitted parameters for reporting
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

from src.config import MARGINALS, OUTPUT_PATHS

# ======================================================
# Load Data
# ======================================================

DATA_PATH = f"{OUTPUT_PATHS['simulations']}synthetic_annual_losses.csv"
df = pd.read_csv(DATA_PATH)

life = df["life_losses"].values
property_ = df["property_losses"].values

# ======================================================
# Utility
# ======================================================

def clip_pit(u, bounds):
    return np.clip(u, bounds[0], bounds[1])


# ======================================================
# Life Insurance Marginal
# ======================================================

def fit_life_marginal(data):
    cfg = MARGINALS["life"]

    if cfg["distribution"] == "lognormal":
        shape, loc, scale = stats.lognorm.fit(data, floc=0)
        dist = stats.lognorm(s=shape, loc=loc, scale=scale)

        params = {
            "distribution": "lognormal",
            "shape": shape,
            "loc": loc,
            "scale": scale
        }

    elif cfg["distribution"] == "gamma":
        shape, loc, scale = stats.gamma.fit(data, floc=0)
        dist = stats.gamma(a=shape, loc=loc, scale=scale)

        params = {
            "distribution": "gamma",
            "shape": shape,
            "loc": loc,
            "scale": scale
        }

    else:
        raise ValueError("Unsupported life marginal distribution")

    u = dist.cdf(data)
    u = clip_pit(u, cfg["clip_pit"])

    return u, params


# ======================================================
# Property Insurance Marginal (Body + Tail)
# ======================================================

def fit_property_marginal(data):
    cfg = MARGINALS["property"]

    threshold = np.quantile(data, cfg["threshold_quantile"])
    body_data = data[data <= threshold]
    tail_data = data[data > threshold] - threshold

    # --- Fit body ---
    body_shape, body_loc, body_scale = stats.lognorm.fit(body_data, floc=0)
    body_dist = stats.lognorm(
        s=body_shape,
        loc=body_loc,
        scale=body_scale
    )

    # --- Fit GPD tail ---
    gpd_shape, gpd_loc, gpd_scale = stats.genpareto.fit(tail_data, floc=0)
    gpd_dist = stats.genpareto(
        c=gpd_shape,
        loc=gpd_loc,
        scale=gpd_scale
    )

    # --- PIT calculation ---
    u = np.zeros(len(data))
    p_body = body_dist.cdf(threshold)

    for i, x in enumerate(data):
        if x <= threshold:
            u[i] = body_dist.cdf(x)
        else:
            u[i] = p_body + (1 - p_body) * gpd_dist.cdf(x - threshold)

    u = clip_pit(u, cfg["clip_pit"])

    params = {
        "threshold": threshold,
        "body": {
            "distribution": "lognormal",
            "shape": body_shape,
            "loc": body_loc,
            "scale": body_scale
        },
        "tail": {
            "distribution": "gpd",
            "shape": gpd_shape,
            "loc": gpd_loc,
            "scale": gpd_scale
        }
    }

    return u, params


# ======================================================
# Main Execution
# ======================================================

if __name__ == "__main__":

    life_u, life_params = fit_life_marginal(life)
    property_u, property_params = fit_property_marginal(property_)

    # Save PIT values
    np.save(f"{OUTPUT_PATHS['simulations']}life_pit.npy", life_u)
    np.save(f"{OUTPUT_PATHS['simulations']}property_pit.npy", property_u)

    # Save parameters
    all_params = {
        "life": life_params,
        "property": property_params
    }

    with open(f"{OUTPUT_PATHS['tables']}marginal_parameters.json", "w") as f:
        json.dump(all_params, f, indent=4)

    print("Marginal fitting and PIT transformation completed.")
