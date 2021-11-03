import numpy as np


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    # Compute the number of samples created

    N = len(samples)
    denom_1 = -1 / (2*h**2)
    denom_2 = 1 / (np.sqrt(2 * np.pi ) * h * N)
    pos = np.arange(-5, 5.0, 0.1)
    estDensity = np.sum(np.exp(denom_1 * ((pos[np.newaxis, :] - samples[:, np.newaxis])**2)), axis = 0) * denom_2
    
    estDensity = np.hstack((pos[:, np.newaxis], estDensity[:, np.newaxis]))
    return estDensity

'''
samples = np.random.normal(0, 1, 100)
h = 0.3
tmp = kde(samples, h)
print(tmp)
'''