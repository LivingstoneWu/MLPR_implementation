import preprocessing
import fitting

'''
Naive Bayes generative classifier assumes that the feature vectors of each class comes from a Gaussian distribution.
These Gaussian distributions are fitted with MLE.

Assume X: feature matrix
yy: label vectors, containing only positive integers indicating classes
'''


def train_naive_bayes(X, yy):
    fitted_params_dict={}
    vals, feature_dict=preprocessing.sep_classes(X, yy)
    for val in vals:
        fitted_params_dict[val]=fitting.mle_gaussian(feature_dict[val])
    return fitted_params_dict