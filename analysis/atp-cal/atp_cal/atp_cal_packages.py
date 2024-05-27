def my_test():
    print("Package is working!")

#################### IMPORTS ####################
    
# Numpy imports:
import numpy as np

#Scipy imports
import scipy
from scipy import optimize
from scipy.optimize import curve_fit

# for extracting filenames
import glob

# skimage submodules we need
import skimage.io
import skimage.measure
import skimage.filters
import skimage.exposure
import skimage.morphology
from skimage.registration import phase_cross_correlation

#Matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Seaborn imports (stylistic for nice plots)
import seaborn as sns
rc={'lines.linewidth': 2, 'axes.labelsize': 14, 'axes.titlesize': 14, \
    'xtick.labelsize' : 14, 'ytick.labelsize' : 14, }
sns.set(style='ticks', rc=rc)

# show images in viridis by default
plt.rcParams['image.cmap'] = 'viridis'

#for DataFrames
import pandas as pd

#To interact with the operating system
import os

#For status bar 
from tqdm.notebook import tqdm as tqdm



#################### FUNCTIONS ####################

    
def file_sorter(file_list):
    """
    Some file names in image sets get mis-ordered as 0, 1, 10, 11, 12, ..., 19, 2, 20, 21, ... 
    This function corrects the ordering to 0, 1, 2, 3, ..., 10, 11, ...
    
    Parameters:
    file_list: list of all the image file names as strings
    
    Returns:
    A list of function names sequentially ordered.
    """
    
    #Extracts and makes a list of all the position numbers
    pos = []
    for file in file_list:
        position = int(file.split('Pos')[-1].split('.ome')[0])
        pos.append(position)

    #Keeps the path strings for before and after the position number
    pre_pos = file.split('Pos')[0] + 'Pos'
    post_pos = '.ome' + file.split('.ome')[-1]

    #Collects the indices of how array gets sorted
    ind_order = np.argsort(pos)

    #Orders the files
    ordered_files = []
    for ind in ind_order:
        ordered_files.append(pre_pos + str(pos[ind]) + post_pos)
    
    return ordered_files



def file_to_image(files):
    """
    Takes in a file list and converts to images.
    
    Parameters:
    files: list of file names that you want to read into images
    
    Returns:
    An array of np.int16 images
    """
    im_list=list()
    for file in files:
        im = skimage.io.imread(file)
        im_list.append(im.astype(np.int16))
    
    return np.array(im_list)


def binary_im_generator(im_for_binary, percentile = 80):
    """
    Creates an image of ones for values above a threshold and zeros for all others. 
    This is essentially a mask. The image is the same size as the inputted image.
    
    Parameters:
    im_for_binary: image you wish to create a mask from
    percentile: uses the np.percentile function which thresholds above 
                the given percentile. Automatically set to 80
                
    Returns:
    A binary image to be used as a mask
    """
    
    im_binary = im_for_binary > np.percentile(im_for_binary, percentile)
    
    #plot the figures
    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(16,8))

    loc0 = ax0.imshow(im_for_binary)
    fig.colorbar(loc0, ax=ax0, shrink = 0.5)
    ax0.set_title("inputted image")
    ax0.grid(False)

    loc1 = ax1.imshow(im_binary, cmap = plt.cm.Greys_r)
    fig.colorbar(loc1, ax=ax1, shrink = 0.5)
    ax1.set_title("binary image")
    ax1.grid(False)
    
    return im_binary


def norm_mat_fn_iATP(im_ref, im_dark, r_blur=3):
    """
    Generate a normalization matrix from a reference image.
    
    This function corrects for uneven illumination. It takes in a reference image which should be the zero 
    ATP control. It then finds the brightest point of the image and for each pixel replaces the pixel value 
    with (brightest pixel value)/(initial pixel value). This creates a matrix that when multiplied by the 
    original image with flatten the illumination and raise every pixel to the brightest value. You can then 
    multiply this normalization matrix by all other images in the dataset to flatten the illumination in each image.

    It is assumed that the reference image is taken for a sample with a spatially uniform protein concentration. The 
    normalization value is everywhere greater than 1, except at the position of the highest illumination.
    
    Parameters
    ----------
    im_ref : numpy array
        Reference image (e.g., the first frame in the MT channel).
    
    offset_camera : float
        Camera offset (not accounting for autofluorescence).
        
    r_blur : float
        Radius of blurring to be performed on the reference image 
        in order to remove noise and short length scale nonuniformities [px].
        
    Returns
    -------
    norm_mat : numpy array
        Normalization matrix
    """
    
    # Convert image into a float type
    im_ref = im_ref.astype(float)
    
    # Subtract the camera offset
    im = im_ref - im_dark
    
    # Rescale the image down for faster denoising
    rescale = 4.0
    im_resized = skimage.transform.rescale(im, 1/rescale)
    
    # Median filter to remove hot pixels
    im_median = skimage.filters.median(im_resized, skimage.morphology.disk(10.0))
    
    # Gaussian blur the image
    im_blur = skimage.filters.gaussian(im_median, r_blur)
    
    # Find the location of the peak
    ind_max = np.where(im_blur == im_blur.max())
    i_center, j_center = ind_max[0][0], ind_max[1][0]
    
    # Peak fluorescence in the normalization image
    fluo_peak = im_blur[i_center, j_center]
    
    # Normalization matrix
    norm_mat = fluo_peak/im_blur 
    
    # Scale up the normalization matrix to the original size
    norm_mat = skimage.transform.rescale(norm_mat, rescale)
    
    return norm_mat


def Langmuir(conc, a, b, c):
    """
    Given a concentration value, this function returns an intensity value based on the Langmuir binding function
    
    Parameters
    conc = 1D array of concentrations
    a, b, c, parameters of the function
    
    Returns
    A 1D array of intensities corresponding to the given concentrations
    """
    return ((b*(conc/a)/(1+(conc/a)))+c)



def Langmuir_curve_fit(conc, calavg, maxconc, p0):
    """
    Performs a curve fitting using scipy.optimize.curve_fit to fit data to a Langmuir curve
    
    Parameters
    conc = 1D array of concentrations
    calavg = 1D array of average intensity values of data
    maxconc = scalar Maximum concentration of data taken
    p0 = 1D list with 3 entries of parameter guesses for a, b, and c in the Langmuir function
    
    Returns
    param = 1D list with fit values of each parameter
    curve = 1D array of intensity values for every concentration in xvals
    xvals = 1D array from 0 to maxconc with step size 1
    """
    
    
    #Curve fits and returns parameter values as well as the covarience
    param, param_cov = curve_fit(Langmuir, conc, calavg, p0) 

    #stores the new function information according to the coefficients given by curve-fit() function 
    xvals=np.linspace(0,maxconc,maxconc)
    curve = Langmuir(xvals, param[0], param[1], param[2])
    
    return param, curve, xvals