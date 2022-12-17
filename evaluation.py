import numpy as np


def mean_sq_error(yy_hat, yy):
    return np.mean((yy_hat - yy) ** 2)
