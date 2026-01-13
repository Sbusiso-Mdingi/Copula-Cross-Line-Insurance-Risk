"""
Synthetic data generation for cross-line insurance risk modelling.

This module generates independent annual aggregate losses for:
- Life insurance (mortality-driven)
- Property insurance (catastrophe-driven)

Dependence is introduced later via copulas.
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.config import OUTPUT_PATHS

# ======================================================
# Global Settings
# ======================================================

np.random.seed(RANDOM_SEED)

N_YEARS = 30  # Typical historical data length

# ======================================================
# Life Insurance Loss Generation
# ======================================================

def generate_life_losses(n_years: int) -> np.ndarray:
    """
    Generate annual aggregate life insurance losses.

    Modelling assumptions:
    - Right-skewed mortality shocks
    - Moderate variance
    - Occasional extreme years
    """

    # Base mortality loss (in millions)
    mu = 50
    sigma = 0.35

    losses = np.random.lognormal(mean=np.log(mu), sigma=sigma, size=n_years)

    return losses


# ======================================================
# Property Insurance Loss Generation
# ======================================================

def generate_property_losses(n_years: int) -> np.ndarray:
    """
    Generate annual aggregate property catastrophe losses.

    Modelling assumptions:
    - Heavy-tailed catastrophe risk
    - Low-frequency, high-severity behaviour
    """

    # Frequency of catastrophe years
    freq = np.random.poisson(lam=0.8, size=n_years)

    losses = []

    for n_events in freq:
        if n_events == 0:
            losses.append(5.0)
        else:
            severities = np.random.pareto(a=2.5, size=n_events) * 40
            losses.append(severities.sum())

    return np.array(losses)


# ======================================================
# Main Execution
# ======================================================

def generate_and_save_data(n_years: int = N_YEARS) -> pd.DataFrame:
    """
    Generate synthetic loss data and save to disk.
    """

    life_losses = generate_life_losses(n_years)
    property_losses = generate_property_losses(n_years)

    data = pd.DataFrame({
        "year": np.arange(1, n_years + 1),
        "life_losses": life_losses,
        "property_losses": property_losses
    })

    output_path = f"{OUTPUT_PATHS['simulations']}synthetic_annual_losses.csv"
    data.to_csv(output_path, index=False)

    return data


if __name__ == "__main__":
    df = generate_and_save_data()
    print("Synthetic loss data generated.")
