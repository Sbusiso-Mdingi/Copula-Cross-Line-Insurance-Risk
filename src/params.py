import numpy as np

params = np.load(
    "output/tables/copula_parameters.npy",
    allow_pickle=True
).item()

print(params)
