import preprocessing
import fitting
import functions
import numpy as np

'''
Naive Bayes generative classifier assumes that the feature vectors of each class comes from a Gaussian distribution.
These Gaussian distributions are fitted with MLE.

Assume X: feature matrix
yy: label vectors, containing only positive integers indicating classes
'''


def train(X, yy):
    fitted_params_dict={}
    vals, feature_dict=preprocessing.sep_classes(X, yy)
    for val in vals:
        fitted_params_dict[val]=fitting.mle_gaussian(feature_dict[val])
    return fitted_params_dict


"""Give prediction of a single sample
Params:
xx: feature vector

Returns:
probs: predicted probabilities of the sample is from each of the classes.
"""


def predict_single(xx, params_dict):
    classes=sorted(list(params_dict.keys()))
    preds=[functions.mv_gaussian(xx, *params_dict[c]) for c in classes]
    return functions.softmax(preds)


"""Give prediction of multiple samples based on the Naive Bayes model
Params:
X: feature matrix containing samples to be predicted

Returns:
preds: ndarray
    matrix of prediction, each row containing probabilities of coming from the corresponding classes
"""


def predict(X, params_dict):
    res=[]
    for row in X:
        res.append(predict_single(row, params_dict))
    return np.array(res)