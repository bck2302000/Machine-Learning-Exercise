import numpy as np
from kern import kern
import cvxopt


def svmkern(X, t, C, p):
    # Non-Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                        (num_samples x dim)
    # t        : labeling                           (num_samples x 1)
    # C        : penalty factor the slack variables (scalar)
    # p        : order of the polynom               (scalar)
    #
    # OUTPUT:
    # sv       : support vectors (boolean)          (1 x num_samples)
    # b        : bias of the classifier             (scalar)
    # slack    : points inside the margin (boolean) (1 x num_samples)

    #####Insert your code here for subtask 2d#####
    
    N = len(X)  # size of data points
    q = cvxopt.matrix(-1 * np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(N), np.eye(N)]))
    A = cvxopt.matrix(t[np.newaxis, :])
    b = cvxopt.matrix(np.double(0))
    LB = np.zeros(N)
    UB = C * np.ones(N)
    h = cvxopt.matrix(np.hstack([-LB, UB]))
    temp = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            temp[i, j] = kern(X[i,:], X[j,:], p)
    H = cvxopt.matrix(temp * (t[:, np.newaxis] @ t[np.newaxis, :]))
    
    sol = cvxopt.solvers.qp(H, q, G, h, A, b)
    alpha = np.array(sol['x']).reshape((-1,))
    sv = np.where(alpha > 1e-6, True, False)
    w = np.sum((alpha[sv] * t[sv]).reshape((-1,1)) * X[sv], axis = 0)
    b = np.mean(t[sv][:,np.newaxis] - X[sv] @ w.reshape(-1,1))
    result = (X @ w.reshape(-1,1) + b).reshape((-1,))
    slack = np.where(alpha > C - 1e-6, True, False)
    
    return alpha, sv, b, result, slack
