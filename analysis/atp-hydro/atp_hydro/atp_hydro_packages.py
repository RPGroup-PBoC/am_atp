def my_test():
    print("Package is working!")

#################### IMPORTS ####################

import time 

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
    
    return np.array(norm_mat)


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
    epsilon = 1e-10  # Small constant to avoid division by zero
    result = a * np.power(((c - array) / (array - b + epsilon)),(1/d), dtype=np.float128)
    result = np.clip(result, np.finfo(result.dtype).min, np.finfo(result.dtype).max)
    return result

#     return a * ((c - array) / (array - b)) ** (1/d)

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

    # print('exponential parameters ', param)
    
    return param, curve

def linearfunc(x, m, c):
    return m*x + c

def linearfit(time, norm_conc):
    # Use curve_fit to fit the linear function to the data
    params, covariance = curve_fit(linearfunc,time, norm_conc)

    return params

def rsqrd(data, fit):
    ssres = np.sum((data - fit)**2)
    sstot = np.sum((data-np.average(data))**2)
    return 1 - ssres/sstot

def analyze_hydrolysis(bound_files, unbound_files, frame_int, skip_int, cal_params, p0, Motconc, bound_bg=1914, unbound_bg=1914):
    """
        Analyzes tiff files

        Input: 
        bound_files:    List of location of tiff files of bound channel.
        unbound_files:  List of location of tiff files of bound channel.
        frame_int:      Frame Interval (in seconds).
        skip_int:       Number of images to skip. Used for large datasets.
        cal_params:     Callibration parameters [Km, Rmax (max ratio), Rmin (min ratio), n (exponent)]
        p0:             Initial guess for fitting exponential curve to data.
        Motconc:        Motor concentration used in experiments.
        bound_bg:       Camera offset in tiff files for bound channel.
        unbound_bg:     Camera offset in tiff files for unbound channel.
    """
    print("start")

    max_index = 100; 

    # Convert files to images and save as array:
    bound_array = file_to_image(bound_files[:max_index])
    unbound_array = file_to_image(unbound_files[:max_index])

    # Subtract background from all calibration images
    bound_bs = bound_array - bound_bg
    unbound_bs = unbound_array - unbound_bg

    # set negative values to zero
    unbound_bs[unbound_bs<0] = 0
    bound_bs[bound_bs<0] = 0
    
    # Find the normalization matrix
    bound_norm_mat = norm_mat_fn_iATP(bound_array[-1], bound_bg)
    unbound_norm_mat = norm_mat_fn_iATP(unbound_array[-1], unbound_bg)
    
    # Normalize all the calibration images by multiplying by the normalization matrix
    bound_norm = bound_bs*bound_norm_mat
    unbound_norm = unbound_bs*unbound_norm_mat
    
    # Average intensities
    bound_hydro = np.mean(bound_norm, axis=(1,2))
    unbound_hydro = np.mean(unbound_norm, axis=(1,2))
    index = min(len(bound_hydro), len(unbound_hydro)); 
    ratio_hydro = bound_hydro[:index]/unbound_hydro[:index]; 

    print("before")
    print(bound_norm.shape)
    # Calculate standard deviation in ratio for each image
    total_ratio = bound_norm[:100, :, :]/unbound_norm[:100, :, :]; 
    ratio_hydro_std = np.std(total_ratio, axis = (1, 2)); 
    print("after")

    # Calculate std in atp for each image
    total_atp = ATP_inten_to_conc(total_ratio, cal_params[0],  cal_params[1],  cal_params[2],  cal_params[3]); 
    atp_std = np.nanstd(total_atp, axis = (1, 2)); 
    
    #define time
    time_array = np.arange(0, len(ratio_hydro), 1)*frame_int*skip_int; 

    #convert ratios to concentration values
    ratio_concavg = ATP_inten_to_conc(ratio_hydro, cal_params[0],  cal_params[1],  cal_params[2],  cal_params[3]); 
    
    #Remove any nans
    #find nans
    nans = np.where(np.isnan(ratio_concavg)==True)

    #remove
    ratio_hydro_uM = np.delete(ratio_concavg, nans)
    times = np.delete(time_array, nans)
    
    # Ratio should have atleast three points
    if len(ratio_hydro_uM) > 3: 

        # Find index after first 2 mins. This is where you will begin fitting.
        two_min_index = np.where(times >= 120)[0][0]; 

        # Find index where ATP conc is max, after first 2 mins.
        start_index = two_min_index + np.where(ratio_hydro_uM[two_min_index:] == np.amax(ratio_hydro_uM[two_min_index:]))[0][0]; 

        # If ATP value falls below certain % of max ATP concentration
        if len(np.where(ratio_hydro_uM[start_index:] <= 0.75*np.amax(ratio_hydro_uM))[0]) != 0:
            # Find index at which ATP conc reaches certain % of max value
            end_index = np.where(ratio_hydro_uM[start_index:] <= 0.75*np.amax(ratio_hydro_uM))[0][0]; 
        else: 
            end_index = len(ratio_hydro_uM) - 1; 

        # to do - remove this if statement? It shouldn't be necessary...
        if start_index + 1 < end_index:
            linear_params = linearfit(times[start_index:end_index], ratio_hydro_uM[start_index:end_index])

            #Find r^2 for linear fit
            linear_r2 = rsqrd(
                ratio_hydro_uM[start_index:end_index], 
                linearfunc(times[start_index:end_index], linear_params[0], linear_params[1])
                )
            
            # Store details of linear fitting
            linear_data_regime = [times[start_index], times[end_index], len(ratio_hydro_uM[start_index:end_index])]; #start time, end time, number of points used.

        else: 
            linear_params = [np.nan, np.nan];  
            linear_r2 = np.nan; 
            linear_data_regime = [np.nan, np.nan, np.nan]; #start time, end time, number of points used.

        # Try exponential fitting, set to nan if fitting fails
        try:
            exponential_fit_start_time = times[start_index]; 
            exp_params, curve = expfit(times[start_index:], ratio_hydro_uM[start_index:], p0)

            if np.all((np.where(ratio_hydro_uM-exp_params[2] <= (exp_params[1]-exp_params[2])/np.e**2)[-1]).size != 0):
                exp_e2 = np.where(ratio_hydro_uM-exp_params[2] <= (exp_params[1]-exp_params[2])/np.e**2)[-1][0]; 
                exp_r2 = rsqrd(np.log(ratio_hydro_uM[:exp_e2]-exp_params[2]), np.log(curve[:exp_e2]-exp_params[2])); 
            else: 
                exp_e2 = np.nan; 
                exp_r2 = np.nan; 
            
            # Compute per motor hydrolysis rate
            ATPsat = exp_params[1]-exp_params[2] #uM
            Decay=exp_params[0] #s
            slope = ATPsat/Decay #uM/s
            rate = ATPsat/Decay/Motconc

        except:
            exp_params = [np.nan, np.nan, np.nan];  
            rate = np.nan;  
            exp_r2 = np.nan;  
            exponential_fit_start_time = np.nan;  

    else: 
        linear_params = [np.nan, np.nan];  
        linear_r2 = np.nan; 
        exp_params = [np.nan, np.nan, np.nan];  
        rate = np.nan;  
        exp_r2 = np.nan;  
        linear_data_regime = [np.nan, np.nan, np.nan]; #start time, end time, number of points used.
        exponential_fit_start_time = np.nan; 

    return linear_params, linear_r2, exp_params, rate, exp_r2, times, ratio_hydro_uM, atp_std, ratio_hydro, ratio_hydro_std, bound_hydro, unbound_hydro, linear_data_regime, exponential_fit_start_time