import numpy as np
def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector 1xK for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : log-likelihood

    #####Insert your code here for subtask 6a#####
    
    D, K = covariances.shape[1:]
    N = X.shape[0]
    logLikelihood = 0
    
    for i in range(N):
        prob = 0
        for j in range(K):
            coeff = 1 / float(((2 * np.pi)**(D/2))*(np.sqrt(np.linalg.det(covariances[...,j]))))
            if len(X.shape) < 2:
                diff = X - np.array(means[j])
            else:
                diff = X[i,:] - np.array(means[j])
            prob += weights[j] * coeff * np.exp(-0.5*diff.T @ np.linalg.inv(covariances[...,j]) @ diff)
        logLikelihood += np.log(prob)
            
    return logLikelihood

