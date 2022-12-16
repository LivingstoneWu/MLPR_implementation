import math

def RBF(xx, cc, h):
    return math.exp(-(xx-cc)**2/h**2)

def sigmoid(a):
    return 1/(1+math.exp(-a))
