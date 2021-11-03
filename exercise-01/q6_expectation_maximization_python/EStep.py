import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each Gaussian DxDxK
    # X              : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.

    #####Insert your code here for subtask 6b#####
    
    D, K = covariances.shape[1:]
    N = X.shape[0]
    logLikelihood = 0
    gamma = np.zeros((N, K))
    
    for i in range(N):
        prob = 0
        for j in range(K):
            coeff = 1 / float(((2 * np.pi)**(D/2))*(np.sqrt(np.linalg.det(covariances[...,j]))))
            diff = X[i,:] - np.array(means[j])
            gamma[i, j] = weights[j] * coeff * np.exp(-0.5*diff[np.newaxis,:] @ np.linalg.inv(covariances[...,j]) @ diff[:,np.newaxis])
            prob += gamma[i,j]
        
        gamma[i,:] = gamma[i,:] / prob
        logLikelihood += np.log(prob)
        
    
    
    return [logLikelihood, gamma]
