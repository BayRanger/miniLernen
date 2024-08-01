# https://medium.com/@bianca_ionescu99/using-pybind11-for-python-bindings-of-c-code-linux-wsl-part-i-e52479c3fb7
# pybind is a header only lib

from SumFunction import *
from numpy_demo2 import *
from numpy_demo1 import *
import numpy as np


print(sum(10,2))

data1 = np.array([1,2,3])
data2 = np.array([1,2,3])
result = add_arrays_1d(data1, data2)

print(result)