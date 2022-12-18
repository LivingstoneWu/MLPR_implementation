import math
import numpy as np

def RBF(xx, cc, h):
    return math.exp(-(xx-cc)**2/h**2)

def sigmoid(a):
    return 1/(1+math.exp(-a))

def list_RBFs(ccs, hs):
    if len(ccs)!=len(hs):
        raise RuntimeError("RBF list: The list of center locations and hs are expected to have same lengths.")
    RBFs=[]
    for i in range(len(ccs)):
        RBFs.append(lambda xx:RBF(xx, ccs[i], hs)[i])
    return RBFs

