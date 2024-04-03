def my_test():
    print("Package is working!")

#################### IMPORTS ####################
    
#for reading files
import glob

#math computation and data organization
import numpy as np
import scipy
import pandas as pd

# For loading bars
from tqdm.notebook import tqdm as tqdm

#For image plotting
import skimage.io

#For identifying aster center
from skimage.filters import threshold_otsu, gaussian, threshold_mean
from skimage.measure import regionprops
import cv2

#for fitting
from lmfit import minimize, Parameters, fit_report

#for image registration
from skimage.registration import phase_cross_correlation
import os

#Matplotlib plotting packages
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import patches
from matplotlib import gridspec

#Movie
import celluloid as cell
import matplotlib.animation as animation

#for saving data
import csv



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

#---------------------------------------------------------------------------------------

def file_to_image(files):
    """
    Takes in a file list and converts to images.
    
    Parameters:
    files: list of file names that you want to read into images
    
    Returns:
    An array of np.int16 images
    """
    im_list = list()
    for file in files:
        im = skimage.io.imread(file)
        im_list.append(im.astype(np.int16))

    return np.array(im_list)

#---------------------------------------------------------------------------------------

def binary_im_generator(im_for_binary, plot=True):
    """
    Uses Otsu Thresholding to create a mask the same size as the inputted image. Pixels above the threshold will be set to 1's and pixels below the threshold will be set to 0's. Otsu thresholding is used due to its ability to separate foreground from background. This is convenient here as ATP images have sharp circular boundaries. 
    
    Parameters:
    im_for_binary: image you wish to create a mask from
    plot=True: if True will plot the original image with 
                
    Returns:
   im_binary: A binary image to be used as a mask
   center: (x,y) values of the circular mask center
   radius: radius of the mask
    """
    
    # Create the binary image:
    im_binary = (im_for_binary > threshold_otsu(im_for_binary)).astype(np.uint8)
    
    # Find the circular contour of the binary image
    contours, hierarchy = cv2.findContours(im_binary, 1, 2)
    for j, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 1:
            max_area = area
            max_index = j
    
    # Save the center and radius values
    (x, y), radius = cv2.minEnclosingCircle(contours[max_index])
    
    # If true plot the figure - original figure with surrounding contour circle
    if plot==True:
         fig,ax=plt.subplots()
         ax.imshow(im_for_binary, vmin=1900, vmax=np.percentile(im_for_binary,99.9))
         circle1 = plt.Circle((x, y), radius, fill=False, color='r')
         ax.add_patch(circle1)
    
    return im_binary, (x,y), radius


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