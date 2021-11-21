import numpy as np
# might need to add path to mingw-w64/bin for cvxopt to work
# import os
# os.environ["PATH"] += os.pathsep + ...
import cvxopt


def svmlin(X, t, C):
    # Linear SVM Classifier
    #
    # INPUT:
    # X        : the dataset                  (num_samples x dim)
    # t        : labeling                     (num_samples x 1)
    # C        : penalty factor for slack variables (scalar)
    #
    # OUTPUT:
    # alpha    : output of quadprog function  (num_samples x 1)
    # sv       : support vectors (boolean)    (1 x num_samples)
    # w        : parameters of the classifier (1 x dim)
    # b        : bias of the classifier       (scalar)
    # result   : result of classification     (1 x num_samples)
    # slack    : points inside the margin (boolean)   (1 x num_samples)


    #####Insert your code here for subtask 2a#####
    N = len(X)  # size of data points
    q = cvxopt.matrix(-1 * np.ones(N))
    G = cvxopt.matrix(np.vstack([-np.eye(N), np.eye(N)]))
    A = cvxopt.matrix(t[np.newaxis, :])
    b = cvxopt.matrix(np.double(0))
    LB = np.zeros(N)
    UB = C * np.ones(N)
    h = cvxopt.matrix(np.hstack([-LB, UB]))
    H = cvxopt.matrix((X @ X.T) * (t[:, np.newaxis] @ t[np.newaxis, :]))
    
    sol = cvxopt.solvers.qp(H, q, G, h, A, b)
    alpha = np.array(sol['x']).reshape((-1,))
    sv = np.where(alpha > 1e-6, True, False)
    w = np.sum((alpha[sv] * t[sv]).reshape((-1,1)) * X[sv], axis = 0)
    b = np.mean(t[sv][:,np.newaxis] - X[sv] @ w.reshape(-1,1))
    result = (X @ w.reshape(-1,1) + b).reshape((-1,))
    slack = np.where(alpha > C - 1e-6, True, False)
    
    return alpha, sv, w, b, result, slack

