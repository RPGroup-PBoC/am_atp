"""
Functions to perform a gliding assay analysis!
"""

#######################################################################################################################

####### Imports #########

#Numpy - for scientific computing
import numpy as np
import scipy

#For image reading and analysis
import glob
import skimage.io
import skimage.filters as sf
from skimage import io, measure, filters, transform, morphology

#Matplotlib plotting packages
import matplotlib as mpl
import matplotlib.pyplot as plt


# For loading bars
from tqdm.notebook import tqdm as tqdm

# For creating DataFrames
import pandas as pd

#######################################################################################################################

####### Functions #########

def my_test():
    print("Package is working!")
    
#------------- Image Processing -------------#

def stackimport(im_dir):#file_path, file_folder, included_frames):
    
    #im_dir = file_path+file_folder+included_frames

    #Import Stack
    im_stack = []
    files = np.sort(glob.glob(im_dir))
    for file in files:
        im_stack.append(skimage.io.imread(file).astype(np.int16))
    im_stack = np.array(im_stack)
    
    return im_stack

def thresh_processing(im):

    #Normalize the image
    image_norm = (im - im.min())/(im.max()-im.min())

    #-----Thresholding-----#
    #Niblack
    thresh_niblack = filters.threshold_niblack(image_norm, window_size= 3, k=0.001)

    #Otsu
    thresh_otsu = filters.threshold_otsu(thresh_niblack)

    #Make a binary image where values above the threshold are = 1 and all else is 0
    im_thresh = (thresh_niblack > thresh_otsu)
    
    return im_thresh


def MTregions(im_thresh):
    
    #make a image with each patch labeled a unique number
    im_label, n_label = measure.label(im_thresh, return_num=True)

    #save the unique labels
    unique_regions, region_counts = np.unique(im_label, return_counts = True)
    
    return im_label, unique_regions, region_counts

##-----Remove non-linear/small/edge tracks-----#
#--- edges ---#
# #give some border width
# edge = 3
def edgeremove(im_label, edge=3):
    #make an image that will have the border removed
    im_border = np.copy(im_label)

    #Make the border mask
    border = np.ones(np.shape(im_label))
    border[edge:-1*edge, edge:-1*edge] -=1

    #Ask if there is a nonzero label within the border region and if so remove that patch
    for n in np.unique(im_border):
        if np.any(border*[im_border==n+1]):
            im_border[im_border==n+1] = 0

    #Save image with border patches eliminated
    im_internal = im_border
    
    return im_internal

#--- small ---#

def smallremove(im_internal, unique_regions, region_counts, area_thresh=20):
    # # Define a size threhshold
    # area_thresh = 20

    #Make a dictionary that connects the region label number and the number of pixels within said region
    dict_area = dict(zip(unique_regions, region_counts))

    #Want to find if each patch is big enough
    im_sized = np.copy(im_internal)
    for label in unique_regions:
      #ask if region is larger than threshold
      if label>0 and dict_area[label]<=area_thresh:
        im_sized[im_sized==label] = 0

    return im_sized

#--- thin ---#
def makeskele(im_sized):
    #create skeletons that preserve connectivity
    im_thinned = morphology.thin(im_sized)

    #Relabel so each skelton has its own label I.D
    im_relabel = measure.label(im_thinned)
    
    return im_relabel

#--- remove not straight lines ---#
def onlylines(im_relabel, padding=3, min_dist=9, min_angle=20):

# # Set scanning parameters
# padding=3 #3
# min_dist=9 #9
# min_angle=20 #75

    # Make a copy of the image
    im_cleaned = im_relabel.copy()

    #Define angles to test lines over
    test_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)

    #Perform the transform for each labeled region
    for n in np.unique(im_relabel)[1:]:
        x,y = np.where(im_relabel==n)
        mt_segment = im_relabel[x.min()-padding:x.max()+padding,y.min()-padding:y.max()+padding]

        h, theta, d = transform.hough_line(mt_segment, theta=test_angles)
        _, angles, _ = transform.hough_line_peaks(h, theta, d, 
                                                min_distance=min_dist,
                                                min_angle=min_angle,
                                                num_peaks=2)

        # If more than one line is fit to the region, eliminate the region from our image
        if len(angles)>=2:
            im_cleaned = np.where(im_cleaned==n, 0, im_cleaned)
        
    return im_cleaned

##---------- Create a dataframe -----------##
def makedf(im_cleaned, im_stack, start, stop):
    im_max = np.amax(im_stack[start:stop], axis=0)
    
    # Determine properties - we will need the centroid, major axis length etc
    im_props = measure.regionprops_table(im_cleaned, 
                                         im_max, 
                                         properties = ['area', 
                                                        'major_axis_length', 
                                                        'minor_axis_length', 
                                                        'centroid', 
                                                        'orientation', 
                                                        'label'])
    #Save as dataframe
    df = pd.DataFrame(im_props)

    #exclude Mt with less than 3 pixels
    df = df[df['area']>3]
    df=df.reset_index()

    return df

## -------- deifine MT track as the pixels that fall along the region-------##
# Define a function
def mt_track(xc, yc, l, theta):
    # Ensure that theta is not greater than pi
    if theta >= np.pi:
        theta = theta - np.pi
  
  # If the track is more horizontal, find the x values for pixels falling along 
  #the line, then determine the corresponding y values
    if np.abs(theta) <= np.pi/4 or np.abs(theta)>3*np.pi/4:
        xvals = np.arange(xc-np.abs(0.5*l*np.cos(theta)), xc + np.abs(0.5*l*np.cos(theta)))
        yvals = ( np.tan(theta)*xvals ) + ( yc - np.tan(theta)*xc )
        
  # If the track is more vertical, find the y values for pixels falling along 
  # the line, then determine the x values
    else:
        yvals = np.arange(yc - np.abs(0.5*l*np.sin(theta)), yc + np.abs(0.5*l*np.sin(theta)))
        xvals = (yvals/np.tan(theta) + (xc - yc/np.tan(theta)))
  
  # Return the x and y values
    return xvals, yvals

## ----- Kymo stuff ----- ##
def kymothresh(kymo, sigma):
  # Create a gaussian blur
    gauss = sf.gaussian(kymo, sigma = sigma)
    kymo_thresh = gauss
  #Threshold
    # kymo_otsu = sf.threshold_otsu(gauss)
    # kymo_thresh = gauss>kymo_otsu
    
    return kymo_thresh

def skeletonfit(kymo_thresh, pixel_size):
    kymo_otsu = sf.threshold_otsu(kymo_thresh)
    kymo_thresh = kymo_thresh>kymo_otsu
    kymo_thin = morphology.thin(kymo_thresh)
    xvals = np.where(kymo_thin)[1]
    yvals = np.where(kymo_thin)[0]
    # points of the skeleton to lines
    mfit_inv, bfit_inv = np.polyfit(yvals, xvals, deg=1)
    mfit = 1/mfit_inv
    bfit = - bfit_inv/mfit_inv

    #Calculate the speed (nm/s)
    speed = 1000*pixel_size/np.abs(mfit)

    #calculate the r^2 value of the fit
    ss_res =  np.sum((yvals - (mfit*xvals+bfit))**2)
    ss_tot = np.sum((yvals - np.average(yvals))**2)
    rsqrd = 1-(ss_res/ss_tot)

    return xvals, yvals, [mfit, bfit], speed, rsqrd

def centroidfit(kymo_thresh, pixel_size):
    kymo_otsu = sf.threshold_otsu(kymo_thresh)
    kymo_thresh = kymo_thresh>kymo_otsu
    
    #Find the center of mass of each row
    xvals = []
    for t, row_vals in enumerate(kymo_thresh):
        xindex = np.arange(len(row_vals))
        com = np.sum(row_vals*xindex)/np.sum(row_vals)
        xvals.append(com)
    xvals = np.array(xvals)

    #Find if there are nans - each time step is 1 second so it is just the index, would otherwise have to multiply by delta t
    yvals = np.where(~np.isnan(xvals))[0]
    xvals = xvals[np.where(~np.isnan(xvals))[0]]

    #Fit a line
    mfit_inv, bfit_inv = np.polyfit(yvals, xvals, deg=1)
    mfit = 1/mfit_inv
    bfit = - bfit_inv/mfit_inv
    #Calculate the speed (nm/s)
    speed = 1000*pixel_size/np.abs(mfit)

    #calculate the r^2 value of the fit
    ss_res =  np.sum((yvals - (mfit*xvals+bfit))**2)
    ss_tot = np.sum((yvals - np.average(yvals))**2)
    rsqrd = 1-(ss_res/ss_tot)

    return xvals, yvals, [mfit, bfit], speed, rsqrd 

def kymo_func(kymo, sigma, pixel_size, plot_returns = False):
    """
  params: kymo: image
          sigma: for the gaussian blur
          pixel_size: convert pixels to um
          method: 
          plot: (boolean) if you want to print the plot

  returns: speed
           rsqrd: r-squared squared value of the fit to the kymograph
           [mfit, bfit]: list of fit parameters
           frames: time values
           row_coms[frames]: center of mass of the MT at each time value

    """
    #Threshold the kymograph
    kymo_thresh = kymothresh(kymo, sigma)
    
    #Fit the kymographs based on centroid method, seems like the more accurate speed detector
    centroid_xvals, centroid_yvals, centroid_params, centroid_speed, centroid_rsqrd = centroidfit(kymo_thresh, pixel_size)

    #Fit the kymographs based on the skeleton method (here we really want to save the r^2 and get rid of anything negative good at picking up "Y"s)
    skele_xvals, skele_yvals, skele_params, skele_speed, skele_rsqrd = skeletonfit(kymo_thresh, pixel_size)
    
    if plot_returns == True:
        return centroid_speed, centroid_rsqrd, skele_rsqrd, centroid_xvals, centroid_yvals, centroid_params
    else:
        return centroid_speed, centroid_rsqrd, skele_rsqrd
    
    
## ----- calculate speeds-----##
def speedcalculator(df, im_stack, start, stop):
    #Create empty lists for the speeds, r-squared values, and the number of centroids for each kymograph
    speeds=[]
    r2 = []
    skeler2 = []

    #Iterate through each MT track
    for i in range(len(df)):

        # Define the line fitting the track
        xc =  df.loc[i, 'centroid-1']
        yc =  df.loc[i, 'centroid-0']
        l =  df.loc[i, 'major_axis_length']
        theta = -df.loc[i, 'orientation'] + (np.pi/2)

        xtrack, ytrack = mt_track(xc, yc, l, theta)

        #Ensure there is more than one point saved in the track line
        if len(xtrack)>1:

          # Create a kymograph
            mykymo = []

            for im in im_stack[start:stop]:
                ls=[]
                for i in range(len(xtrack)):
                    if int(ytrack[i])<512 and int(xtrack[i])<512:
                        ls.append(im[int(ytrack[i]), int(xtrack[i])])
                mykymo.append(ls)
            mykymo = np.array(mykymo)

            # Compute the fit to the kymograph
            speed, centroid_rsqrd, skele_rsqrd = kymo_func(mykymo, sigma = 2, pixel_size = 0.161)

            #Append the fit results to the empty lists
            speeds.append(speed)
            r2.append(centroid_rsqrd)
            skeler2.append(skele_rsqrd)

        #if there is fewer than two centroids detected in the kymograph drop the track
        else:
            df=df.drop([i])

    # Include the new columns in the DataFrame
    speeds = np.array(speeds)
    r2 = np.array(r2)
    skeler2 = np.array(skeler2)
    df['speed (nm/s)'] = speeds
    df['r^2'] = r2
    df['skele r^2'] = skeler2
    
    return df

##---- save good tracks ----##
def goodtracks(df, r2thresh=0.8, skelethresh=0):
    df_good = df[df['r^2']>r2thresh][df['skele r^2']>skelethresh]
    return df_good


### ---------- MT Lengths -----------###
def cropped_MT_isolate(im):
    norm = (im - im.min())/(im.max()-im.min())
    im_niblack = filters.threshold_niblack(norm, window_size= 3, k=0.001)
    im_otsu = filters.threshold_otsu(im_niblack)
    
    #Make a binary image where values above the threshold are = 1 and all else is 0
    im_thresh = (im_niblack > im_otsu)
    
    #make a image with each patch labeled a unique number
    MT_label, num_label = measure.label(im_thresh, return_num=True)
    
    # Determine properties - we will need the centroid, major axis length etc
    MT_props = measure.regionprops_table(MT_label, im, properties = ['area', 
                                                    'major_axis_length', 
                                                    'minor_axis_length', 
                                                    'centroid', 
                                                    'orientation', 
                                                    'label'])

    return MT_label, MT_props

## --- are points in ellipse ----- ##
def inellipse(x, y, x0, y0, major_l, minor_l, orient):
    major_len = 0.5*major_l
    minor_len=0.5*minor_l
    orientation = -orient+np.pi/2
    checkvals = (((x-x0)*np.cos(orientation) + (y-y0)*np.sin(orientation))**2 / major_len**2) + (((x-x0)*np.sin(orientation) - (y-y0)*np.cos(orientation))**2 / minor_len**2)
    return checkvals<=1

## 
def MTlenths(df_good, im_stack, start, stop, umperpixel):
    
    df_newind=df_good.reset_index()
    
    #Store each MT length
    MT_lengths=[]

    #iterate through tracks
    for index in df_newind['level_0']:
        #define track parameters
        xc =  df_good.loc[index, 'centroid-1']
        yc =  df_good.loc[index, 'centroid-0']
        lmajor =  df_good.loc[index, 'major_axis_length']
        lminor =  df_good.loc[index, 'minor_axis_length']
        theta = -df_good.loc[index, 'orientation'] + (np.pi/2)

        #Find the track axis lengths
        xmajor, ymajor = mt_track(xc, yc, lmajor, theta)
        xminor, yminor = mt_track(xc, yc, lminor, theta-np.pi/2)

        #define a crop window
        crop_win = int(lmajor/2+1)
        crop = np.s_[int(yc-crop_win) : int(yc+crop_win+1), int(xc-crop_win) : int(xc+crop_win+1)]
        if int(xc-crop_win)<0:
            crop = np.s_[int(yc-crop_win) : int(yc+crop_win+1), 0 : int(xc+crop_win+1)]
        if int(yc-crop_win)<0:
            crop = np.s_[0 : int(yc+crop_win+1), int(xc-crop_win) : int(xc+crop_win+1)]

        #Define the axis in the cropped reference frame
        cropmajx = xmajor-xc + crop_win 
        cropmajy = ymajor-yc + crop_win 

        #Find the length of the microtuble
        lengths = []

        #Create an ellipse fit for each MT region
        for j, im in enumerate(im_stack[start:stop]):
            im_label, im_props = cropped_MT_isolate(im[crop])
            for labi in np.unique(im_label)[:-1]:
                #Determine if the track line is in the MT ellipse
                inellip = inellipse(cropmajx, 
                                    cropmajy, 
                                    im_props['centroid-1'][labi], 
                                    im_props['centroid-0'][labi], 
                                    im_props['major_axis_length'][labi], 
                                    im_props['minor_axis_length'][labi], 
                                    im_props['orientation'][labi])
                #If MT track is in the ellipse, add the major axis length to the array
                if np.sum(inellip)>1:
                    lengths.append(im_props['major_axis_length'][labi])

        #average the lengths and convert the to ums
        MT_lengths.append(np.average(lengths)*umperpixel)
        
    df_good['MT len (um)'] = MT_lengths
    return df_good

#--------- Add columns for concentrations ------------#

def expparam_adder(df_good, file, start, acqu_interval):
    
    #extract relevant concentrations/times etc.
    data_mot = float(file.split('/')[-1].split('_')[2].split('uM')[0])
    data_MT = float(file.split('/')[-1].split('_')[3].split('MT')[0])
    data_ATP = float(file.split('/')[-1].split('_')[4].split('uM')[0])
    data_ADP = float(file.split('/')[-1].split('_')[5].split('uM')[0])
    data_P = float(file.split('/')[-1].split('_')[6].split('uM')[0])
    data_time = float(file.split('/')[-1].split('_')[7].split('min')[0])*60 + (start*acqu_interval)

    #Add parameters to the dataframe
    zeros = np.zeros(len(df_good))

    df_good['motor conc'] = zeros + data_mot
    df_good['MT dilute'] = zeros + data_MT
    df_good['ATP'] = zeros + data_ATP
    df_good['ADP'] = zeros + data_ADP
    df_good['P'] = zeros + data_P
    df_good['time (seconds)'] = zeros + data_time
    
    return df_good