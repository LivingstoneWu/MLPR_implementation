import numpy as np

def add_gaussian(yy, noise, use_seed=False, random_seed=0):
    if use_seed:
        rs=np.random.RandomState(seed=random_seed)
    else:
        rs=np.random.RandomState()
    return yy+rs.normal(0, noise, yy.shape[0])


def lin1d_with_gaussian(w_mean=0, w_var=2, b_mean=0, b_var=2, noise=0.5, num_points=20, interval=(-10, 10), use_seed=False, random_seed=0):
    if use_seed:
        rs=np.random.RandomState(seed=random_seed)
    else:
        rs=np.random.RandomState()
    xx=rs.rand(num_points)*(interval[1]-interval[0])+interval[0]
    w=rs.normal(w_mean, w_var)
    b=rs.normal(b_mean, b_var)
    yy=xx @ w + b
    return xx, add_gaussian(yy, noise, use_seed, random_seed)


