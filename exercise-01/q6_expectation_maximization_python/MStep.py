import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    
    N, K = gamma.shape
    D = X.shape[1]
    N_hat = np.sum(gamma, axis = 0)
    weights = N_hat / N
    
    means = (gamma.T @ X) / N_hat[:, np.newaxis]
    
    covariances = np.zeros((D,D,K))
    for i in range(K):
        temp = np.zeros((D,D))
        for j in range(N):
            diff = X[j] - means[i]
            temp += gamma[j, i] * (diff[:, np.newaxis] @ diff[np.newaxis, :])
        covariances[..., i] = temp / N_hat[i]
        
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return weights, means, covariances, logLikelihood
