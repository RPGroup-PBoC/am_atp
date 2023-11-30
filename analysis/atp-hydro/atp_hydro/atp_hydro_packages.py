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
    'xtick.labelsize' : 14, 'ytick.labelsize' : 14}
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

def ATP_inten_to_conc(array, a, b, c, d):
    return a * ((c - array) / (array - b)) ** (1/d)

def expfunc(time, tau, Ao, Ainf):
    return (Ao-Ainf)*np.exp(-time/tau) + Ainf
def expfit(time, norm_conc, p0):
    #Curve fits and returns parameter values as well as the covarience
    param, param_cov = curve_fit(expfunc, 
                                 time, 
                                 norm_conc, 
                                 p0, 
                                 bounds = (np.zeros(3), np.ones([3])*np.inf))

    #stores the new function information according to the coefficients given by curve-fit() function 
    curve = expfunc(time, param[0], param[1], param[2])
    
    return param, curve

def rsqrd(data, fit):
    ssres = np.sum((data - fit)**2)
    sstot = np.sum((data-np.average(data))**2)
    return 1 - ssres/sstot

def analyze_hydrolysis(bound_files, unbound_files, frame_int, skip_int, cal_params, p0, Motconc, bound_bg=1914, unbound_bg=1914):
    """
    test
    """
    
    # Convert files to images and save as array:
    bound_array = file_to_image(bound_files)
    unbound_array = file_to_image(unbound_files)

    #Subtract background from all calibration images
    bound_bs = bound_array - bound_bg
    unbound_bs = unbound_array - unbound_bg

    #set negative values to zero
    unbound_bs[unbound_bs<0] = 0
    bound_bs[bound_bs<0] = 0
    
    #Find the normilization matrix
    bound_norm_mat = norm_mat_fn_iATP(bound_array[-1], bound_bg)
    unbound_norm_mat = norm_mat_fn_iATP(unbound_array[-1], unbound_bg)
    
    #Normalize all the calibration images by multiplying by the normalization matrix
    bound_norm = bound_bs*bound_norm_mat
    unbound_norm = unbound_bs*unbound_norm_mat
    
    #Average intensities
    bound_hydro = np.average(bound_norm, axis=(1,2))
    unbound_hydro = np.average(unbound_norm, axis=(1,2))
    ratio_hydro = bound_hydro/unbound_hydro
    
    #define time
    time = np.arange(0, len(ratio_hydro), 1)*frame_int*skip_int #s

    #convert ratios to concentration values
    ratio_concavg = ATP_inten_to_conc(ratio_hydro, cal_params[0],  cal_params[1],  cal_params[2],  cal_params[3])
    
    #Remove any nans
    #find nans
    nans = np.where(np.isnan(ratio_concavg)==True)

    #remove
    ratio_hydro_uM = np.delete(ratio_concavg, nans)
    times = np.delete(time, nans)
    
    #Fit the exponential
    params, curve = expfit(times, ratio_hydro_uM, p0)
    
    #Find r^2
    e2 = np.where(ratio_hydro_uM-params[2] <= (params[1]-params[2])/np.e**2)[-1][0]
    r2 = rsqrd(np.log(ratio_hydro_uM[:e2]-params[2]), np.log(curve[:e2]-params[2]))
    
    # Compute per motor hydrolysis rate
    ATPsat = params[1]-params[2] #uM
    Decay=params[0] #s
    slope = ATPsat/Decay #uM/s
    rate = ATPsat/Decay/Motconc
    
    return params, rate, r2