import numpy as np
from models import lin_reg
import data_gen
import matplotlib.pyplot as plt
import preprocessing
import functions

xx, yy=data_gen.lin1d_with_gaussian(noise=0.5)
print(len(xx))
plt.plot(xx, yy, "rx")
xgrid=np.linspace(-10, 10, 100)
ccs=np.arange(-10, 10.1, 0.5)
hs=np.array([0.5]*41)
basis_func=functions.list_RBFs(ccs, hs)
basis_func.append(lambda x:x)

reg_factor=0.5
weights=lin_reg.train(preprocessing.preproc_X(xx, basis_func, reg_factor), preprocessing.preproc_yy(yy, basis_func, reg_factor))
yy_hat=preprocessing.preproc_X(xgrid, basis_func)@weights
plt.plot(xgrid, yy_hat)
plt.show()

# plt.plot(xgrid, np.vectorize(basis_func[0])(xgrid))
# plt.show()