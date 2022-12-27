import math
import numpy as np


def RBF(xx, cc, h):
    return math.exp(-(xx - cc) ** 2 / h ** 2)


def sigmoid(a):
    return 1 / (1 + math.exp(-a))


def list_RBFs(ccs, hs):
    if len(ccs) != len(hs):
        raise RuntimeError("RBF list: The list of center locations and hs are expected to have same lengths.")
    RBFs = []
    for i in range(len(ccs)):
        RBFs.append(lambda xx, i=i: RBF(xx, ccs[i], hs[i]))
    return RBFs


def gaussian(x, mean, var):
    return 1 / math.sqrt(2 * math.pi) / var * math.exp(-(x - mean) ** 2 / var ** 2 / 2)


def mv_gaussian(xx, mean, cov):
    return math.pow(math.pi * 2, -len(xx) / 2) / np.sqrt(np.linalg.det(cov)) * np.exp(
        -1 / 2 * ((xx - mean).T @ np.linalg.inv(cov) @ (xx - mean)))


# optimized likelihood function for multiple samples of multivariate gaussian distribution
def likelihood_mv_gaussian(X, mean, cov, log=True):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    coe = math.pow(math.pi * 2, -X.shape[1] / 2) / np.sqrt(det)
    if log:
        neglogsum = X.shape[0] * np.log(coe)
        for row in X:
            neglogsum += 1 / 2 * ((row - mean).T @ inv @ (row - mean))
        return neglogsum
    else:
        res = 1
        for row in X:
            res *= coe * np.exp(-1 / 2 * ((row - mean).T @ inv @ (row - mean)))


def softmax(xx):
    xx=[math.exp(x) for x in xx]
    esum=sum(xx)
    return np.array([x/esum for x in xx])
