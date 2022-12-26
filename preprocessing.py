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


def preproc_X(X, basis_funcs=None, reg_factor=0):
    if basis_funcs is not None:
        res = []
        for row in X:
            row_res = []
            for basis_f in basis_funcs:
                row_res.append(basis_f(row))
            res.append(row_res)
        res = np.array(res)
    else:
        res = np.array(X)
    res = np.concatenate((res, np.ones(res.shape[0])[:, np.newaxis]), axis=-1)
    if reg_factor != 0:
        res = np.concatenate((res, reg_factor * np.identity(res.shape[1])), axis=0)
    return res


"""Preprocess yy vector

"""


def preproc_yy(yy, basis_funcs, reg_factor=0):
    if reg_factor != 0:
        yy = np.concatenate((yy, np.zeros(len(basis_funcs) + 1)))
    return yy


"""Divide the dataset by classes
X: feature matrix
yy: label vector, containing only positive integers indicating discrete classes

returns:
vals: sorted list of discrete labels
res_dict: dictionary containing label: matrix of feature vectors corresponding to the label
"""


def sep_classes(X, yy):
    idx_sort = np.argsort(yy)
    vals, idx_start, counts = np.unique(yy[idx_sort], return_index=True, return_counts=True)
    res_dict = {}
    for i in range(len(vals)):
        res_dict[vals[i]] = X[idx_sort[idx_start[i]:idx_start[i] + counts[i]], :]
    return sorted(vals), res_dict
