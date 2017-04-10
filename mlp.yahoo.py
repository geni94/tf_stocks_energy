import numpy as np
import tensorflow as tf
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

pth = filePath + 'yahoostock.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1,4))
A = scale(A)

y = A[:, 1].reshape(-1, 1)

A = A[:, 0].reshape(-1, 1)

mpl.plot(A[:, 0], y[:, 0])
mpl.show()
