import numpy as np
from scipy.optimize import minimize
import preprocessing


def linreg_cost(ww, Phi, yy):
    return np.sum((Phi @ ww - yy) ** 2)


def linreg_predict(X, ww, basis_funcs):
    Phi=preprocessing.preproc_X(X, basis_funcs)
    return Phi @ ww





"""Trains a linear regression model and returns the corresponding weight vector.

Analytical solution: ww=(Phi^T Phi)^{-1}Phi^T*yy. Or optionally return the result obtained with gradient-based optimizer.

    Parameters
    ----------
    Phi: ndarray
        transformed input feature matrix
    yy: ndarray
        transformed output vector
    grad_opt: boolean, optional
        optimize with gradient optimizer if set True

"""


def train(Phi, yy, grad_opt=False):
    if not grad_opt:
        return np.linalg.lstsq(Phi, yy, rcond=None)[0]
    else:
        return minimize(linreg_cost, np.zeros(Phi.shape[1]),(Phi, yy), "L-BFGS-B", jac=True, options={'maxiter': 500, 'disp': False})