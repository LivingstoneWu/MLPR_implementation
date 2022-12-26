from scipy.optimize import minimize
import numpy as np
import functions

"""Helper function for wrapping and unwrapping parameters to feed into minimize()
Flattens the parameters list, returns a function params_unwrap for recovering desired parameters
"""


def params_unwrap(params, shapes, sizes):
    params_rebuilt = []
    idx = 0
    for i in range(len(shapes)):
        params_rebuilt.append(np.reshape(params[idx:idx + sizes[i]], shapes[i]))
        idx += sizes[i]
    return params_rebuilt


def params_wrap(params):
    params_list = [np.array(param) for param in params]
    shapes = [param.shape for param in params_list]
    sizes = [param.size for param in params_list]
    params_flatterned = np.array()
    for param in params_list:
        params_flatterned = np.concatenate((params_flatterned, param.flatten()))
    params_unwrap = lambda params: params_unwrap(params, shapes, sizes)
    return params_flatterned, params_unwrap


""" Fitting a Gaussian distribution by MLE
Params:
X: matrix containing sample vectors

Returns:

"""


def mle_gaussian(X):
    guess_mean = np.mean(X, axis=0)
    guess_cov = np.cov(X)
    args, params_unwrap = params_wrap([guess_mean, guess_cov])
    target_f = lambda vec: functions.likelihood_mv_gaussian(X, *params_unwrap(vec))
    return params_unwrap(minimize(target_f, args))
