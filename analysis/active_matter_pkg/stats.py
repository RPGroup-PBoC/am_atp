"""
The stats package includes functions for different statistical
analyses and visualization methods.
"""
import numpy as np
from scipy import optimize
from skimage import filters

def hpd(trace, mass_frac) :
    """
    Returns highest probability density region given by
    a set of samples. Taken from Justin Bois's bebi103
    repository.

    Parameters
    ----------
    trace : array
        1D array of MCMC samples for a single variable
    mass_frac : float with 0 < mass_frac <= 1
        The fraction of the probability to be included in
        the HPD.  For example, `massfrac` = 0.95 gives a
        95% HPD.
        
    Returns
    -------
    output : array, shape (2,)
        The bounds of the HPD
    """
    # Get sorted list
    d = np.sort(np.copy(trace))

    # Number of total samples taken
    n = len(trace)
    
    # Get number of samples that should be included in HPD
    n_samples = np.floor(mass_frac * n).astype(int)
    
    # Get width (in units of data) of all intervals with n_samples samples
    int_width = d[n_samples:] - d[:n-n_samples]
    
    # Pick out minimal interval
    min_int = np.argmin(int_width)
    
    # Return interval
    return np.array([d[min_int], d[min_int+n_samples]])

def avg_profile_df(df, metric, independent, x_min=0, x_max=200, n_positions=100, dx=1.5, avg_method='median'):
    """
    Determines the average profile for 'metric' as a function of 'independent' from a DataFrame
    -----------------
    df : DataFrame containing metric and independent variable of interest
    metric : str, column of df which averaged profile 
    independent : str, independent variable column in df
    """
    x_intervals = np.linspace(x_min, x_max, n_positions)

    avg_vals = np.zeros(n_positions)
    std_vals = np.zeros(n_positions)

    for n, x in enumerate(x_intervals):
        dat = df[(df[independent] > x-0.5*dx) & (df[independent] < x+0.5*dx)][metric]
        if avg_method == 'median':
            avg_vals[n] = np.median(dat)
        elif avg_method == 'mean':
            avg_vals[n] = np.mean(dat)
        else:
            raise ValueError('Incorrect averaging method specified.')
        std_vals[n] = np.std(dat)

    return x_intervals, avg_vals, std_vals

def avg_profile(y, x, x_min=0, x_max=200, n_positions=100, dx=1.5, avg_method='median'):
    x_intervals = np.linspace(x_min, x_max, n_positions)

    avg_vals = np.zeros(n_positions)
    std_vals = np.zeros(n_positions)

    for n, x_ in enumerate(x_intervals):
        dat = y[(x > x_-0.5*dx) & (x < x_+0.5*dx)]
        if avg_method == 'median':
            avg_vals[n] = np.median(dat)
        elif avg_method == 'mean':
            avg_vals[n] = np.mean(dat)
        else:
            raise ValueError('Incorrect averaging method specified.')
        std_vals[n] = np.std(dat)

    return x_intervals, avg_vals, std_vals

def correlation_optflo(r, v_x, v_y, r_min=0, r_max=200, n_positions=100, dr=2):
    # Compute the correlation in the optical flow vector field as a function of radius
    r_intervals = np.linspace(r_min, r_max, n_positions)

    mag = np.sqrt(v_x**2 + v_y**2)
    u_x = v_x / mag
    u_y = v_y / mag

    correlation = np.zeros(len(r_intervals))

    u_x0 = np.mean(u_x[(r > r_min - dr / 2) & (r < r_min + dr / 2)])
    u_y0 = np.mean(u_y[(r > r_min - dr / 2) & (r < r_min + dr / 2)])
    denominator = (u_x0**2 + u_y0**2)

    for n,radius in enumerate(r_intervals):

        x_comp = u_x0 * u_x[(r > radius - dr / 2) & (r < radius + dr / 2)]
        y_comp = u_y0 * u_y[(r > radius - dr / 2) & (r < radius + dr / 2)]
        numerator = np.mean(x_comp + y_comp)
        
        correlation[n] = numerator / denominator

    return r_intervals, correlation

def profile_fn(im, r_min=0, dr=1.5, avg_method='median', n_positions=100):
    
    """
    Calculate the radial intensity profile of an image, 
    assuming that the (0,0) coordinate is at the center of the image.
    
    Parameters
    ----------
    im : numpy array
        The cropped image of the aster (grayscale).
      
    r_min : float
        Minimum radius beyond which the intensity profile is calculated.
    
    dr : float
        Radial binning size [pixels].
    
    avg_method : string
        The method used for doing a polar average of intensities.
        'median' - median averaging
        'mean' - mean averaging
        
    n_positions : integer
        Number of uniformly spaced radial positions where the average
        intensity is calculated.
    
    Returns
    -------
    r_unif_ls : numpy array
        Uniformly spaced radii where the average intensity is evaluated.
        
    avg_ls : numpy array
        Average intensities evaluated at 'n_positions' different uniformly
        spaced radii.
        
    std_ls : numpy array
        Standard deviations of intensitiy values in differents radial bins
        of size 'dr'.
        
    r_ls : numpy array
        Radial distances of all image pixels from the image center.
        
    im_ls : numpy array
        All pixel intensities in the same order as 'r_ls'.
    """
    
    # Ensure that the image is a numpy array
    if not isinstance(im, np.ndarray):
        im = np.array(im)
    
    # Dimensions of the image
    H,W = im.shape
        
    # Array of the radial distance of each point from the center
    x_mat, y_mat = np.meshgrid(np.arange(W), np.arange(H))
    r_mat = np.sqrt((x_mat - 0.5*W - 0.5)**2 + (y_mat - 0.5*H - 0.5)**2)
    
    # Convert 2d arrays into 1d arrays
    r_ls = r_mat.flatten()
    im_ls = im.flatten()
    
    # Uniformly spaced radii, starting with r_min and
    # ending with the smallest dimension of the image halved
    # (if the image is square, r_max is half of the side length)
    r_max = 0.5*np.min([H,W])-0.5
    r_unif_ls = np.linspace(r_min, r_max, n_positions)
    
    # Lists to store average and standard deviation values
    # for each radial position
    avg_ls = np.zeros(n_positions)
    std_ls = np.zeros(n_positions)

    for i, r in enumerate(r_unif_ls):
        dat = np.sort(im_ls[(r_ls > r-0.5*dr) & (r_ls < r+0.5*dr)])
        if avg_method == 'median':
            avg_ls[i] = np.median(dat)
        elif avg_method == 'mean':
            avg_ls[i] = np.mean(dat)
        else:
            raise ValueError('Incorrect averaging method specified.')
        std_ls[i] = np.std(dat)
        
    # Keep only the points inside the disk
    im_ls = im_ls[r_ls <= r_max]
    r_ls = r_ls[r_ls <= r_max]
        
    return r_unif_ls, avg_ls, std_ls, r_ls, im_ls

def find_center(im_motor, im_MT, w_half, r_thresh, r_blur, least_squares=True, n_search_half=5, \
                n_positions=30, debug=False, use_max=True):
    """
    Identify the center coordinates of the aster.
    
    Parameters
    ----------
    im_motor : numpy array
        Fluorescence image of motors where the highest fluorescence intensity
        is in the aster region, i.e. bright spots/objects in the background
        are removed.
        
    im_MT : numpy array
        Fluorescence image of microtubules.
        
    w_half : float
        Half the size of the window the points in which will be considered for 
        center identification.
        
    r_thresh : float
        Minimal distance of points from the center used in center identification.
        
    r_blur : float
        Size of Gaussian blurring for initial guessing of the center.
    
    least_squares : Boolean
        Indicator variables that is true is the least squares optimization is
        used (faster), and is false if a window search is used (slower)
        
    n_search_half : int
        Half the size of the window around the initial guess where the center
        is looked for.
        
    n_positions : int
        Number of positions considered when extracting radial profiles.
        
        
    Returns
    -------
    i_center : int
        i-coordinate of identified center
    
    j_center : int
        j-coordinate of identified center
    """
    
    # Apply a Gaussian blur
    im_MT_blur = filters.gaussian(im_MT, r_blur)
    im_motor_blur = filters.gaussian(im_motor, r_blur)

    if use_max:
        # Find the location of the peak
        i_max_MT, j_max_MT = np.where(im_MT_blur == np.max(im_MT_blur))
        i_guess_MT = i_max_MT[0]
        j_guess_MT = j_max_MT[0]
        
        i_max_motor, j_max_motor = np.where(im_motor_blur == np.max(im_motor_blur))
        i_guess_motor = i_max_motor[0]
        j_guess_motor = j_max_motor[0]
    
        if np.sqrt((i_guess_motor - i_guess_MT)**2 + (j_guess_motor - j_guess_MT)**2) < 30:
            i_guess = i_guess_motor
            j_guess = j_guess_motor
        else:
            i_guess = i_guess_MT
            j_guess = j_guess_MT
        if (i_guess - w_half < 0) or (i_guess + w_half > im_MT.shape[0]) or \
            (j_guess - w_half < 0) or (j_guess + w_half > im_MT.shape[1]):
            raise ValueError('Brightest location in the image is near the edge, likely outside the aster.')

    else:
        i_guess = im_MT.shape[0]/2
        j_guess = im_MT.shape[1]/2    
    
    def find_sd(location):
        """
        Calculate the array of standard deviations for a profile
        centered at 'location'.
        """
        # Current coordinates of the aster center
        i_curr, j_curr = location
        
        # Nearest integer coordinates
        ii = [int(i_curr), int(i_curr)+1]
        jj = [int(j_curr), int(j_curr)+1]
        
        # Array to store the average cropped images,
        # obtained by areal weighting of 4 contributions
        im_crop = np.zeros([2*w_half+1,2*w_half+1])
        w_tot = 0
        for i in ii:
            for j in jj:
                weight = (1-np.abs(i_curr-i))*(1-np.abs(j_curr-j))
                w_tot += weight
                im_crop += weight*im_motor[i-w_half:i+w_half+1, j-w_half:j+w_half+1]
                
        _, _, std_ls, _, _ = profile_fn(im_crop, r_thresh, n_positions = n_positions)
        return std_ls
    
    if least_squares:
        param_init = [i_guess, j_guess]
        output = optimize.least_squares(find_sd, param_init, method="lm")
        i_center, j_center = output.x
        i_center = int(round(i_center))
        j_center = int(round(j_center))
    else:
        search_range = np.arange(-n_search_half, n_search_half)
        di_m, dj_m = np.meshgrid(search_range, search_range)
        std_min = 1e5
        di_opt = 0
        dj_opt = 0

        for i in range(len(search_range)):
            for j in range(len(search_range)):
                i_curr = i_max + di_m[i,j]
                j_curr = j_max + dj_m[i,j]

                im_crop = im_motor[i_curr-w_half:i_curr+w_half+1, j_curr-w_half:j_curr+w_half+1]
                _, _, std_ls, _, _ = profile_fn(im_crop, r_thresh)

                if np.mean(std_ls) < std_min:
                    di_opt = di_m[i,j]
                    dj_opt = dj_m[i,j]
                    std_min = np.mean(std_ls)
        
        i_center = i_guess + di_opt
        j_center = j_guess + dj_opt
    
    if debug:
        return i_center, j_center, i_guess, j_guess, im_motor_blur

    return i_center, j_center

def compute_strains(v_x,v_y,sigma=None):
    if sigma!=None:
        v_x = filters.gaussian(v_x,sigma=sigma)
        v_y = filters.gaussian(v_y,sigma=sigma)

    Dxx, dxy = np.gradient(v_x)
    dyx, Dyy = np.gradient(v_y)

    Dxy = (dxy + dyx) / 2
    return Dxx, Dxy, Dyy