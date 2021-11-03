import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import misc
from parameters import parameters
from getLogLikelihood import getLogLikelihood
from estGaussMixEM import estGaussMixEM
from EStep import EStep
from MStep import MStep
from regularize_cov import regularize_cov
from plotModes import plotModes

epsilon, K, n_iter, skin_n_iter, skin_epsilon, skin_K, theta = parameters()

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def skinDetection(ndata, sdata, K, n_iter, epsilon, theta, img):
    # Skin Color detector
    #
    # INPUT:
    # ndata         : data for non-skin color
    # sdata         : data for skin-color
    # K             : number of modes
    # n_iter        : number of iterations
    # epsilon       : regularization parameter
    # theta         : threshold
    # img           : input image
    #
    # OUTPUT:
    # result        : Result of the detector for every image pixel

    #####Start Subtask 1g#####
    print('creating GMM for non-skin')
    weight_nonskin, means_nonskin, cov_nonskin = estGaussMixEM(ndata, K, n_iter, epsilon)
    print('GMM for non-skin completed')
    print('creating GMM for skin')
    weight_skin, means_skin, cov_skin = estGaussMixEM(sdata, K, n_iter, epsilon)
    print('GMM for skin completed')

    height, width, _ = img.shape

    noSkin = np.ndarray((height, width))
    skin = np.ndarray((height, width))

    for h in range(height):
        for w in range(width):
            noSkin[h, w] = np.exp(
                getLogLikelihood(means_nonskin, weight_nonskin, cov_nonskin, np.array([img[h, w, 0], img[h, w, 1],
                                                                                       img[h, w, 2]])))
            skin[h, w] = np.exp(
                getLogLikelihood(means_skin, weight_skin, cov_skin, np.array([img[h, w, 0], img[h, w, 1],
                                                                              img[h, w, 2]])))


    # calculate ration and threshold
    result = skin / noSkin
    result = np.where(result > theta, 1, 0)
    #####End Subtask#####
    return result

sdata = np.loadtxt('skin.dat')
ndata = np.loadtxt('non-skin.dat')

img = im2double(imageio.imread('faces.png'))

skin = skinDetection(ndata, sdata, skin_K, skin_n_iter, skin_epsilon, theta, img)
plt.imshow(skin)
plt.show()