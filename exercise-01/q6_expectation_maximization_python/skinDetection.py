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

    #####Insert your code here for subtask 1g#####
    height, width, _ = img.shape
    nskin_weights, nskin_means, nskin_covariances = estGaussMixEM(ndata, K, n_iter, epsilon)
    skin_weights, skin_means, skin_covariances = estGaussMixEM(sdata, K, n_iter, epsilon)
    prob_nskin = np.zeros((height, width))
    prob_skin = np.zeros((height, width))
    for h in range(height):
        for w in range(width):        
            prob_nskin[h,w] = np.exp(getLogLikelihood(nskin_means, nskin_weights, nskin_covariances, np.array([img[h,w,0], img[h,w,1], img[h,w,2]])))
            prob_skin[h,w] = np.exp(getLogLikelihood(skin_means, skin_weights, skin_covariances, np.array([img[h,w,0], img[h,w,1], img[h,w,2]])))
    result = np.where((prob_skin / prob_nskin) > theta, 1, 0)
    return result

sdata = np.loadtxt('skin.dat')
ndata = np.loadtxt('non-skin.dat')

img = im2double(imageio.imread('faces.png'))

skin = skinDetection(ndata, sdata, skin_K, skin_n_iter, skin_epsilon, theta, img)
plt.imshow(skin)
plt.show()