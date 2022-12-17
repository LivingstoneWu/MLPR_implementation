import numpy as np

"""Generate preprocessed input matrix Phi and output vector yy. 

Phi=[phi(xx_i)]
Generate preprocessed input matrix Phi, with each row containing the result of application of basis functions on the
corresponding xx sample. Optional with regularization terms.

    Arguments
    ----------
    X: ndarray
        original feature matrix
    basis_funcs: list
        list of vectorized basis functions to be applied on the feature matrix.
    reg_factor: float, optional
        L2 regularization penalty factor lambda. 
    
    Returns
    ----------
    ndarray
        a ndarray of the transformed feature matrix Phi.
"""


def preproc_X(X, basis_funcs, reg_factor=0):
    res = []
    for row in X:
        row_res = []
        for basis_f in basis_funcs:
            row_res.append(basis_f(row))
        res.append(row_res)
    res = np.array(res)
    if reg_factor != 0:
        res = np.concatenate((res, np.identity(res.shape[1])), axis=0)
    return res

"""Preprocess yy vector

"""


def preproc_yy(yy, basis_funcs, reg_factor=0):
    if reg_factor!=0:
        yy=np.concatenate((yy, np.zeros(len(basis_funcs))))
    return yy
