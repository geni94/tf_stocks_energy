import tensorflow as tf
import numpy as np
from TFMLP import MLPR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

# from sklearn import datasets
# from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

pth = filePath + 'dbdata.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 4))
A = scale(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)
#Plot the high value of the stock price
mpl.plot(A[:, 0], y[:, 0])
mpl.show()
