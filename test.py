import numpy as np
import pandas as pd

data = pd.read_csv("Data/forestfires.data", index_col=0, header=0).to_numpy()
w = np.load('Weights/forestfires/2/weights.npz')
weights = [w[key] for key in w]
train = data
solutions = train[:,-1]
a = train[:, :-1]
for idx, i in enumerate(weights[:-1]):
    try:
        z = np.dot(a, i)
        if z.ndim > 3:
            raise Exception
    except:
        z = np.einsum('ijk,jkl -> ijl', a, i)
    a = np.tanh(z)
try:
    z = np.dot(a, weights[-1])
    if z.ndim > 3:
        raise Exception
except:
    z = np.einsum('ijk,jkl -> ijl', a, weights[-1])
