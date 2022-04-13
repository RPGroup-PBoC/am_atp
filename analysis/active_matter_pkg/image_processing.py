"""
The image_processing package is designed to process images for identifying
microtubules (in TIRF) or for observing aster formation.
"""
import numpy as np
from skimage import filters, morphology, measure, transform
from scipy import optimize
import pandas as pd

def filter_mts(image, block_size=5, mask_size=5, yen=False):
    image_norm = (image - image.min()) / (image.max() - image.min())

    thresh_niblack = filters.threshold_niblack(image_norm, window_size=block_size,
                                                k=0.001)

    # Rather than applying the threshold to the image to create a binary
    # image, the threshold array thresh_niblack thickens the MTs, reducing
    # filament break-up. This is used then in the Otsu thresholding to
    # produce the binary image.
    thresh_otsu = filters.threshold_otsu(thresh_niblack)
    im_thresh = (thresh_niblack > thresh_otsu)

    mask = morphology.square(mask_size)
    im_closed = morphology.closing(im_thresh, selem=mask)

    if yen==True:
        im_subt = image - im_closed
        im_yen = filters.threshold_yen(im_subt)
        im_filtered = im_subt > im_yen
    else:
        im_filtered = im_closed.copy()

    return im_filtered

def border_clear(im_label, edge=3):
    # Remove objects too close to the border
    im_border = np.copy(im_label)

    border = np.ones(np.shape(im_label))
    border[edge:-1*edge,edge:-1*edge] -= 1

    for n in np.unique(im_border):
        if np.any(border * [im_border==n+1]):
            im_border[im_border==n+1] = 0

    return im_border

def determine_count_nums(im_label):
    """
    Obtains maximum number of objects in the labeled image. Used to determine
    if background subtraction and thresholding must be performed on top of Niblack
    thresholding scheme.
    """
    unique, counts = np.unique(im_label, return_counts=True)

    return unique, counts

def remove_small(im_label, area_thresh=10):
    im_sized = np.copy(im_label)

    unique, counts = determine_count_nums(im_label)

    # Create dictionary except for 0 (background)
    dict_area = dict(zip(unique,counts))

    for label in unique:
        if label > 0 and dict_area[label]<=area_thresh:
            im_sized[im_sized==label] = 0
    
    return im_sized

def remove_large(im_label, area_thresh=10):
    im_sized = np.copy(im_label)

    unique, counts = determine_count_nums(im_label)

    # Create dictionary except for 0 (background)
    dict_area = dict(zip(unique,counts))

    for label in unique:
        if label > 0 and dict_area[label]>=area_thresh:
            im_sized[im_sized==label] = 0
    
    return im_sized

def remove_circulars(im_label, eccen_thresh=0.8):
    im_eccen = im_label.copy()

    im_props = measure.regionprops_table(im_eccen,
                                        properties=['label','eccentricity'])
    df = pd.DataFrame(im_props)

    for n in np.unique(im_eccen):
        if df[df['label']==n]['eccentricity'].values < eccen_thresh:
            im_eccen[im_eccen==n] = 0

    return im_eccen

def are2lines(mt_segment, min_dist=9, min_angle=75):
    """
    Determine if putative microtubules are two microtubules. Uses
    Hough straight lines to determine if there are at least 2
    lines that can be drawn from the putative filament.
    
    Input
    -------
    mt_segment : (M, N), ndarray; cropped region about the putative
                 microtubule
    min_angle : int, minimum angle (in degrees) separating lines (default 75)
    
    Return
    -------
    len(angles)==2 : bool, determines whether there is a crossover
    """
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = transform.hough_line(mt_segment, theta=test_angles)

    _, angles, _ = transform.hough_line_peaks(h, theta, d, 
                                                min_distance=min_dist,
                                                min_angle=min_angle,
                                                num_peaks=2)

    return len(angles)==2

def remove_line_crossovers(image, min_dist=9, min_angle=75, padding=3):
    """
    Removes microtubules that cross over in the images. 
    Input
    -------
    image : (M, N), ndarray; image from which MT crossovers are removed
    min_angle : int, minimum angle (in degrees) separating lines (default 30)
    padding : int, padding around cropped MT segments

    Return
    -------
    im_cleaned : (M, N), ndarray; image with MT crossovers removed
    """
    im_cleaned = image.copy()

    for n in np.unique(image)[1:]:
        x,y = np.where(image==n)
        mt_segment = image[x.min()-padding:x.max()+padding,y.min()-padding:y.max()+padding]
        
        if are2lines(mt_segment, min_dist=min_dist, min_angle=min_angle):
            im_cleaned = np.where(im_cleaned==n, 0, im_cleaned)

    return im_cleaned

def process_mt_images(image, block_size=3, mask_size=3, count_thresh=200, edge=3, area_thresh=10, min_dist=9, min_angle=75, padding=3):
    im_filtered = filter_mts(image, block_size=block_size, mask_size=mask_size)
    im_label, n_labels = measure.label(im_filtered, return_num=True)
    # Determine if Yen thresholding background subtraction is necessary
    unique, _ = determine_count_nums(im_label)
    if unique[-1] > count_thresh:
        im_filtered = filter_mts(image, block_size=block_size, mask_size=mask_size, yen=True)
        im_label, n_labels = measure.label(im_filtered, return_num=True)
    im_internal = border_clear(im_label, edge=edge)
    im_sized = remove_small(im_internal, area_thresh=area_thresh)
    im_thinned = morphology.thin(im_sized)
    im_relabel = measure.label(im_thinned)
    im_noxovers = remove_line_crossovers(im_relabel, min_dist=min_dist,
                                        min_angle=min_angle, padding=padding)

    return im_noxovers

def gaussian_2d(coords, mu_x, mu_y, sigma_x, sigma_y, coeff, offset=0):
    """
    Defines a 2D Gaussian function

    Input
    -------
    coords : 2xN array; x and y coordinates along the Gaussian
    mu_x, mu_y : float; x and y coordinates of Gaussian peak
    sigma_x, sigma_y : float; standard deviations of Gaussian
    coeff : float; coefficient in front of the exponential term
    offset : float; offset value if Gaussian doesn't go to 0. Default 0

    Return
    -------
    coeff * np.exp(-(x - mu_x)**2/(2*sigma_x**2) - (y - mu_y)**2/(2*sigma_y**2))
    """
    exponent = - (coords[:,0] - mu_x)**2/(2*sigma_x**2) - (coords[:,1] - mu_y)**2/(2*sigma_y**2)
    return offset + coeff * np.exp(exponent)

def gaussian_fit(image, alpha_guess, sigma_xguess=20, sigma_yguess=20):
    y = np.arange(np.shape(image)[1], step=1)
    x = np.arange(np.shape(image)[0], step=1)
    Y, X = np.meshgrid(y,x)
    coords = np.column_stack((np.ravel(X), np.ravel(Y)))

    p0 = [np.shape(image)[0]/2, np.shape(image)[1]/2, sigma_xguess, sigma_yguess, alpha_guess]
    popt, pcov = optimize.curve_fit(gaussian_2d, coords, np.ravel(image), p0=p0)

    return popt, pcov

def process_aster_cells(image, sigma=100, small_area_thresh=20, area_thresh=20):
    """
    Performs post processing on post-photobleached cells during aster 
    network formation. Cleans background with Gaussian blurring before
    applying mean thresholding and removing small objects.

    Input
    -------
    image : (M, N), ndarray; raw image to be processed
    sigma : float; sigma on Gaussian filter, default is 100 pixels
    small_area_thresh : int; threshold number of pixels to be large enough to keep
    area_thresh : int; second area thresholding value based on regionprops

    Return
    -------
    im_relabel : (M, N), ndarray; image composed of integer values for each aster unit cell
    regprops : list of RegionProperties
    """
    im_blur = filters.gaussian(image, sigma=sigma)
    im_subt = image - (2**16 - 1) * im_blur
    im_subt[im_subt < 0] = 0

    thresh_mean = filters.threshold_mean(im_subt)
    im_binary = (im_subt > thresh_mean)
    im_binary = morphology.remove_small_objects(im_binary, 20)

    # Try on mean thresholding scheme
    im_labels, num_labels = measure.label(im_binary, return_num=True)
    regionprops = measure.regionprops_table(im_labels, image,
                                            properties=('area','bbox','centroid',
                                                        'convex_area','convex_image',
                                                        'coords','eccentricity',
                                                        'equivalent_diameter','euler_number',
                                                        'extent','filled_area','filled_image',
                                                        'image','inertia_tensor','inertia_tensor_eigvals',
                                                        'intensity_image','label','local_centroid',
                                                        'major_axis_length','max_intensity','mean_intensity',
                                                        'min_intensity','minor_axis_length','moments',
                                                        'moments_central','moments_hu','moments_normalized',
                                                        'orientation','perimeter',
                                                        'slice','solidity','weighted_centroid',
                                                        'weighted_local_centroid','weighted_moments',
                                                        'weighted_moments_central','weighted_moments_hu',
                                                        'weighted_moments_normalized'))
    df_regionprops = pd.DataFrame(regionprops)

    for index in range(1,num_labels+1):
        if df_regionprops[df_regionprops['label']==index]['area'].values[0] <= area_thresh:
            im_labels[im_labels==index] = 0

    im_relabel = measure.label(im_labels)
    regprops = measure.regionprops_table(im_relabel, image,
                                            properties=('area','bbox','centroid',
                                                        'convex_area','convex_image',
                                                        'coords','eccentricity',
                                                        'equivalent_diameter','euler_number',
                                                        'extent','filled_area','filled_image',
                                                        'image','inertia_tensor','inertia_tensor_eigvals',
                                                        'intensity_image','label','local_centroid',
                                                        'major_axis_length','max_intensity','mean_intensity',
                                                        'min_intensity','minor_axis_length','moments',
                                                        'moments_central','moments_hu','moments_normalized',
                                                        'orientation','perimeter',
                                                        'slice','solidity','weighted_centroid',
                                                        'weighted_local_centroid','weighted_moments',
                                                        'weighted_moments_central','weighted_moments_hu',
                                                        'weighted_moments_normalized'))
    df_regprops = pd.DataFrame(regprops)

    return im_relabel, df_regprops

def normalize(im):
    # Sets renormalization of values in NxM array to be between 0 and 1
    return (im - im.min()) / (im.max() - im.min())

def image_mask(image, sigma=30, hw=8):
    im_bk = filters.gaussian(image, sigma=sigma) * (2**16-1)
    im_subt = image - im_bk
    im_subt[im_subt<0] = 0

    thresh_mean = filters.threshold_mean(im_subt)
    im_thresh = (im_subt > thresh_mean)

    # thresholding background image to help remove off-target areas
    thresh_bk = filters.threshold_mean(im_bk)
    im_bk_thresh = (im_bk > thresh_bk)
    im_thresh = im_bk_thresh * im_thresh

    im_binary = morphology.remove_small_objects(im_thresh)
    im_label, n_label = measure.label(im_binary, return_num=True)
    im_border = border_clear(im_label,edge=hw)
    return (im_border>0)

def crop_flow_field(image, u, v, hw=8, sigma=30):

    x = np.arange(0, np.shape(image)[0], 1)
    y = np.arange(0, np.shape(image)[1], 1)

    Y_im, X_im = np.meshgrid(y,x)
    im_mask = image_mask(image, sigma=sigma, hw=hw)

    for n_x in range(image.shape[0]):
        for n_y in range(image.shape[1]):
            x_cent = X_im[n_x,n_y]
            y_cent = Y_im[n_x,n_y]

            x_low = max(0, int(x_cent-hw))
            x_high = min(int(x_cent+hw), image.shape[0])
            y_low = max(0,int(y_cent-hw))
            y_high = min(int(y_cent+hw), image.shape[1])

            window = np.s_[x_low:x_high,y_low:y_high]
            if 1 not in im_mask[window]:
                u[n_x,n_y] = np.nan
                v[n_x,n_y] = np.nan

    return u, v

def mask_value(M, im_mask, num=False):
    if np.shape(M) != np.shape(im_mask):
        raise ValueError('input array and mask image are not the same shape')
    
    M_masked = M.copy()

    if num:
        M_masked[im_mask==0] = 0
    else:
        M_masked[im_mask==0] = np.nan

    return M_masked

def create_window(x,y,winsize):
    """
    Creates a new image window to crop a larger original image
    
    Inputs :
    -----------
    x,y : floats, centers of the image

    Returns :
    -----------
    winsize : float, half-width of window
    """
    return np.s_[int(y-winsize):int(y+winsize),int(x-winsize):int(x+winsize)]