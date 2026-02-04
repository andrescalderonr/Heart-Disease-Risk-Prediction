import numpy as np

w = np.load("w_best.npy")
b = np.load("b_best.npy")[0]
mean = np.load("mean.npy")
std = np.load("std.npy")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(x):
    x = np.array(x).reshape(1, -1)
    x = (x - mean) / std
    z = x @ w + b
    return float(sigmoid(z))
