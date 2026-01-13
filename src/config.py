"""
Global configuration file for cross-line tail risk modelling.

This file centralises all modelling assumptions to ensure:
- Reproducibility
- Transparency
- Easy scenario analysis

No business logic should be placed here.
"""

# ======================================================
# General Settings
# ======================================================

RANDOM_SEED = 123
N_SIMULATIONS = 250_000      # High for tail stability
CONFIDENCE_LEVELS = [0.99, 0.995]

# ======================================================
# Lines of Business
# ======================================================

LINES = ["life", "property"]

# ======================================================
# Marginal Distribution Settings
# ======================================================

MARGINALS = {
    "life": {
        "distribution": "lognormal",
        "fit_method": "MLE",
        "clip_pit": (1e-6, 1 - 1e-6)
    },
    "property": {
        "body_distribution": "lognormal",
        "tail_distribution": "gpd",
        "threshold_quantile": 0.95,
        "fit_method": "POT",        
        "clip_pit": (1e-6, 1 - 1e-6)
    }
}

# ======================================================
# Copula Settings
# ======================================================

COPULAS = {
    "independence": {
        "type": "independence"
    },
    "gaussian": {
        "type": "gaussian",
        "initial_corr": 0.25
    },
    "student_t": {
        "type": "student_t",
        "initial_corr": 0.25,
        "initial_df": 5,
        "df_bounds": (2.1, 30)
    },
    "gumbel": {
        "type": "gumbel",
        "initial_theta": 1.5,
        "theta_bounds": (1.0, 10.0)
    }
}

# ======================================================
# Tail Dependence Reporting
# ======================================================

TAIL_DEPENDENCE = {
    "compute": True,
    "report_upper_tail": True,
    "quantile": 0.99
}

# ======================================================
# Capital & Risk Metrics
# ======================================================

RISK_METRICS = {
    "var": True,
    "tvar": True,
    "tvar_alpha": 0.99,
    "diversification_benefit": True
}

# ======================================================
# Output Paths
# ======================================================

OUTPUT_PATHS = {
    "figures": "output/figures/",
    "tables": "output/tables/",
    "simulations": "output/simulations/"
}
