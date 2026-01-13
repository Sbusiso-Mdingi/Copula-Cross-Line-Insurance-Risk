"""
Copula fitting and joint uniform simulation.

Copulas implemented:
- Independence
- Gaussian
- Student-t
- Gumbel (upper-tail dependence)

Inputs:
- PIT-transformed marginal data

Outputs:
- Fitted copula parameters
- Simulated joint uniforms
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from src.config import COPULAS, N_SIMULATIONS, OUTPUT_PATHS

# ======================================================
# Load PIT Data
# ======================================================

U_LIFE = np.load(f"{OUTPUT_PATHS['simulations']}life_pit.npy")
U_PROP = np.load(f"{OUTPUT_PATHS['simulations']}property_pit.npy")

U = np.column_stack([U_LIFE, U_PROP])

# ======================================================
# Utility
# ======================================================

def gaussian_log_likelihood(rho, u):
    if not -0.99 < rho < 0.99:
        return np.inf

    z = stats.norm.ppf(u)
    cov = [[1, rho], [rho, 1]]

    return -np.sum(stats.multivariate_normal(
        mean=[0, 0],
        cov=cov
    ).logpdf(z))


# ======================================================
# Gaussian Copula
# ======================================================

def fit_gaussian(u):
    res = minimize(
        lambda x: gaussian_log_likelihood(x[0], u),
        x0=[0.2],
        bounds=[(-0.99, 0.99)]
    )
    return {"rho": res.x[0]}


def simulate_gaussian(params, n):
    cov = [[1, params["rho"]], [params["rho"], 1]]
    z = np.random.multivariate_normal([0, 0], cov, size=n)
    return stats.norm.cdf(z)


# ======================================================
# Student-t Copula
# ======================================================

def student_t_log_likelihood(params, u):
    rho, df = params

    if not (-0.99 < rho < 0.99 and df > 2):
        return np.inf

    z = stats.t.ppf(u, df)
    cov = [[1, rho], [rho, 1]]

    return -np.sum(stats.multivariate_normal(
        mean=[0, 0],
        cov=cov
    ).logpdf(z))


def fit_student_t(u):
    res = minimize(
        lambda x: student_t_log_likelihood(x, u),
        x0=[0.2, 5],
        bounds=[(-0.99, 0.99), (2.1, 30)]
    )
    return {"rho": res.x[0], "df": res.x[1]}


def simulate_student_t(params, n):
    cov = [[1, params["rho"]], [params["rho"], 1]]
    z = np.random.multivariate_normal([0, 0], cov, size=n)
    u = stats.t.cdf(z, df=params["df"])
    return u


def student_t_upper_tail_dependence(rho, df):
    return 2 * stats.t.cdf(
        -np.sqrt((df + 1) * (1 - rho) / (1 + rho)),
        df + 1
    )


# ======================================================
# Gumbel Copula
# ======================================================

def gumbel_log_likelihood(theta, u):
    if theta < 1:
        return np.inf

    u1, u2 = u[:, 0], u[:, 1]
    log_u1 = -np.log(u1)
    log_u2 = -np.log(u2)

    A = (log_u1 ** theta + log_u2 ** theta) ** (1 / theta)
    C = np.exp(-A)

    part1 = C * (log_u1 * log_u2) ** (theta - 1)
    part2 = A ** (2 - 2 * theta)
    part3 = theta - 1 + A ** theta

    density = part1 * part2 * part3 / (u1 * u2)
    return -np.sum(np.log(density))


def fit_gumbel(u):
    """
    Fit Gumbel copula parameter theta using MLE.

    Args:
        u (np.ndarray): Nx2 PIT data

    Returns:
        dict: {'theta': theta, 'upper_tail_dependence': lambda_U}
    """
    from scipy.optimize import minimize

    # Log-likelihood function with stability
    def gumbel_log_likelihood(theta_val, data):
        if theta_val <= 1.0:  # enforce lower bound
            return np.inf

        u1, u2 = data[:, 0], data[:, 1]

        # Avoid log(0) issues
        u1 = np.clip(u1, 1e-10, 1 - 1e-10)
        u2 = np.clip(u2, 1e-10, 1 - 1e-10)

        log_u1 = -np.log(u1)
        log_u2 = -np.log(u2)

        A = (log_u1 ** theta_val + log_u2 ** theta_val) ** (1 / theta_val)
        C = np.exp(-A)

        # Avoid zero density
        density = C * (log_u1 * log_u2) ** (theta_val - 1) * A ** (2 - 2 * theta_val) * (theta_val - 1 + A ** theta_val) / (u1 * u2)
        density = np.clip(density, 1e-20, None)

        return -np.sum(np.log(density))

    # Use higher initial guess
    res = minimize(
        lambda x: gumbel_log_likelihood(x[0], u),
        x0=[2.0],  # start above 1 to encourage tail detection
        bounds=[(1.01, 10.0)]
    )

    theta = res.x[0]
    lambda_U = 2 - 2 ** (1 / theta)

    return {"theta": theta, "upper_tail_dependence": lambda_U}



def simulate_gumbel(params, n):
    theta = params["theta"]
    w = np.random.exponential(size=n)
    e1 = np.random.exponential(size=n)
    e2 = np.random.exponential(size=n)

    u1 = np.exp(-(e1 / w) ** (1 / theta))
    u2 = np.exp(-(e2 / w) ** (1 / theta))

    return np.column_stack([u1, u2])


def gumbel_upper_tail_dependence(theta):
    return 2 - 2 ** (1 / theta)


# ======================================================
# Main Execution
# ======================================================

if __name__ == "__main__":

    results = {}

    for name, cfg in COPULAS.items():
        print(f"Fitting {name} copula...")

        if cfg["type"] == "independence":
            results[name] = {"note": "No parameters"}

        elif cfg["type"] == "gaussian":
            params = fit_gaussian(U)
            results[name] = params

        elif cfg["type"] == "student_t":
            params = fit_student_t(U)
            params["upper_tail_dependence"] = student_t_upper_tail_dependence(
                params["rho"], params["df"]
            )
            results[name] = params

        elif cfg["type"] == "gumbel":
            params = fit_gumbel(U)
            params["upper_tail_dependence"] = gumbel_upper_tail_dependence(
                params["theta"]
            )
            results[name] = params

    np.save(f"{OUTPUT_PATHS['tables']}copula_parameters.npy", results)
    print("Copula fitting completed.")