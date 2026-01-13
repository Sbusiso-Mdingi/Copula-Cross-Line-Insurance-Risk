# Cross-Line Tail Risk Modelling Using Copulas

An actuarial risk modelling project analyzing tail dependence between life insurance mortality losses and property catastrophe claims using copulas. The project demonstrates how dependence assumptions materially affect aggregate loss distributions, diversification benefits, and economic capital estimation, with implementation in Python.

This work focuses on joint extreme risk, a core challenge in multi line insurance portfolios and solvency frameworks.

---

## 🚀 Project Overview

Insurers typically write multiple lines of business and rely on diversification to reduce overall risk. A common simplifying assumption is that losses across lines, such as life insurance and property insurance, are independent or weakly correlated.

However, systemic extreme events (e.g. natural catastrophes, pandemics, large-scale disasters) can simultaneously:
- Increase mortality related claims
- Trigger large property catastrophe losses

This project explores how copula based dependence modelling, particularly tail dependence, alters the insurer’s aggregate risk profile compared to traditional independence or Gaussian correlation assumptions.

---

## 🧠 Modelling Objectives

The project is designed around the following actuarial objectives:

- Model cross line dependence between life and property insurance losses
- Capture tail dependence during extreme loss events
- Compare dependence structures and their impact on risk
- Quantify diversification effects under different assumptions
- Estimate economic capital using joint loss simulations

---

## 🧮 Insurance Lines Considered

### Life Insurance
- Annual aggregate mortality driven losses
- Focus on excess mortality during extreme events
- Right skewed loss behaviour

### Property Insurance
- Annual aggregate catastrophe driven claims
- Heavy tailed severity driven by extreme events
- Strong exposure to tail risk

Losses are modelled on an annual aggregate basis, consistent with capital and solvency assessments.

---

## 🔗 Dependence Modelling with Copulas

Copulas are used to model the dependence structure independently of the marginal loss distributions.

The following copula families are fitted and compared:

| Copula | Key Characteristics |
|------|---------------------|
| Gaussian | No tail dependence (baseline comparator) |
| Student t | Symmetric upper and lower tail dependence |
| Gumbel | Upper tail dependence (extreme loss clustering) |

This allows explicit testing of how **tail dependence assumptions** affect joint risk.

---

## 📊 Marginal Modelling

Marginal distributions are fitted independently for each line of business:

- Life insurance losses: Gamma / Lognormal
- Property losses: Heavy tailed models with EVT based tail treatment

Probability integral transforms (PIT) are used prior to copula fitting, with diagnostics including:
- QQ plots
- Goodness of fit tests
- Tail behaviour checks

> Copula results are only as reliable as the marginal models, this project explicitly validates marginals before introducing dependence.

---

## 📈 Risk & Capital Analysis

Using simulated joint loss distributions, the project computes:

- Aggregate annual loss distributions
- Value at Risk (VaR)
- Tail Value at Risk (TVaR)
- Capital comparisons under:
  - Independence
  - Gaussian copula
  - Tail dependent copulas

The analysis highlights how independence assumptions can materially understate economic capital when tail dependence is present.

---

## 🧩 Modelling Pipeline

High level workflow:

Marginal loss modelling → Probability integral transform → Copula fitting → Joint loss simulation → Capital estimation → Actuarial interpretation

Each step is modular and reproducible.

---

## 🖥️ Key Features

- Cross line insurance risk aggregation
- Tail dependent copula modelling
- Joint loss simulation framework
- Capital and diversification analysis
- Clean, production style Python structure
- Reproducible results and diagnostics

---

## 📁 Project Structure

copula-cross-line-insurance-risk/
├── src/
│   ├── config.py
│   ├── data_generation.py
│   ├── eda.py
│   ├── marginals.py
│   ├── copulas.py
│   ├── simulation.py
│   ├── risk_metrics.py
│   └── visualisation.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── output/
│   ├── figures/
│   └── tables/
├── report/
├── README.md
└── requirements.txt

---

## 🧰 Tech Stack

- Python
- NumPy
- Pandas
- SciPy
- Statsmodels
- Copulas (`copulas` library + custom implementations)
- Matplotlib / Seaborn

---

## 📊 Limitations

- Dependence is assumed static over time
- Loss data are simulated but calibrated to realistic insurance magnitudes
- Extreme event history is necessarily limited

These limitations are explicitly acknowledged and discussed, consistent with actuarial best practice.

---

## 🔮 Future Work

Potential extensions include:

- Time varying copulas
- Climate stress and scenario analysis
- Multi line vine copula structures
- Capital allocation by line (Euler principle)
- Reinsurance structure analysis

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio demonstration purposes**.  
It does not constitute actuarial advice and should not be used for real world pricing or capital decisions without appropriate validation.

---

## 👨‍💻 Author

**Sbusiso Mdingi**
