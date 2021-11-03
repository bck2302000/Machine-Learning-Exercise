import numpy as np


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

    #####Insert your code here for subtask 5b#####
    # Compute the number of the samples created

    N = len(samples)
    pos = np.arange(-5, 5.0, 0.1)
    dis = np.sort(np.abs(pos[np.newaxis, :] - samples[:, np.newaxis]), axis = 0)[k - 1, :]
    dis = dis[:, np.newaxis]
    estDensity = k / (N * 2 * dis)
    estDensity = np.hstack((pos[:, np.newaxis], estDensity))
    return estDensity


samples = np.random.normal(0, 1, 100)
k = 30
tmp = knn(samples, k)
print(tmp)
