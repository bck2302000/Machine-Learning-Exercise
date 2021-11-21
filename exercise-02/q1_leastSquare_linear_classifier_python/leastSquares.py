import numpy as np


def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)

    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    N = len(data)
    X_tilda = np.hstack([np.ones(N)[:, np.newaxis], data])
    W_tilda = np.linalg.lstsq(X_tilda.T @ X_tilda, X_tilda.T)[0] @ label[:,np.newaxis]
    weight = W_tilda[1:]
    bias = W_tilda[0]
    return weight, bias


train = {}
test = {}
train.update({'data': np.loadtxt('lc_train_data.dat')})
train.update({'label': np.loadtxt('lc_train_label.dat')})
weight, bias = leastSquares(train['data'], train['label'])
print(weight)
print(bias)