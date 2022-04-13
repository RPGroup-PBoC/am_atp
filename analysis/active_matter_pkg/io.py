"""
This subpackage is designed to tidy up datasets, such as raw image
directories with unnecessary `images`."""

import os
import pandas as pd
import numpy as np

def rm_false_images(directory, keywords=['False Trigger', 'All Off']):
    """
    Removes false images within the directory based on filename. 
    Standard keywords are False Trigger and All Off. Make sure to
    only remove those filenames that do not contain useful images.
    """
    file_list = os.listdir(directory)
    # Keep metadata
    file_list = [item for item in file_list if item[-4:]=='.tif']
    
    for item in file_list:
        if any(list_item in item for list_item in file_list):
            os.remove(os.path.join(directory, file_list, item))

def find_all_tiffs(root_directory):
    """
    Finds all .tif files in root_directory and lists paths. Does not
    work if .tif files are further embedded in subdirectories within
    root_directory
    """
    tiff_list = []

    for root_path, _, files in os.walk(root_directory):
        for f in files:
            if f.endswith('.tif'):
                tiff_list.append(os.path.join(root_path,f))

    return tiff_list

def tiff_walk(root_directory, parse_channels=True):
    """
    Walks through all subdirectories within directory to obtain .tif
    files. Returns list of .tif files in full path. .tif files should
    be at most two directories deep if 'directory' is the root photobleach
    data root: (1) a subdirectory contains different image acquisitions sessions
    for the same (slide,lane,position) combination but the initial aster
    formation process is a separate image acquisition from the photobleaching
    acquisition session; (2) a subdirectory that divides the .tif types
    into channels and cropped images.
    """
    tiff_list = []

    directory_name = os.path.split(root_directory)[-1]

    original_channels = ['DLPRed','DLPYellow','DLP_Red_trimmed','DLP_Yellow_trimmed']

    # Check if keyword found in image acquisition file exists in 'directory'. Here,
    # choice of 'intervals' is chosen as keyword
    if 'intervals' not in directory_name:
        subdirectory = [d.name for d in os.scandir(str(root_directory)) if (directory_name in d.name and os.path.isdir(d.path))]
        directory = [d.path for d in os.scandir(str(root_directory)) if (directory_name in d.name and os.path.isdir(d.path))]
    else:
        subdirectory = []
        directory = [root_directory]
    
    for direct in directory:
        tiff_directory = [d.path for d in os.scandir(str(direct)) if d.name in original_channels]
        for d in tiff_directory:
            tiff_list.extend(find_all_tiffs(d))

    if parse_channels:
        mt_imgfiles = np.sort([imfile for imfile in tiff_list if '/DLPRed/' in imfile and 'DLP_Red_000.tif' in imfile])
        mot_imgfiles = np.sort([imfile for imfile in tiff_list if '/DLPYellow/' in imfile and 'DLP_Yellow_000.tif' in imfile])
        mt_trimmed = np.sort([imfile for imfile in tiff_list if 'DLP_Red_trimmed/' in imfile and '.tif' in imfile])
        mot_trimmed = np.sort([imfile for imfile in tiff_list if 'DLP_Yellow_trimmed/' in imfile and '.tif' in imfile])

        return mt_imgfiles, mot_imgfiles, mt_trimmed, mot_trimmed, subdirectory
    else:
        return tiff_list, subdirectory

def parse_filename(data_directory):
    if 'skip' in data_directory:
        num_skipstr = data_directory.find('skip')
        num_uscore = num_skipstr + data_directory[num_skipstr:].find('_')
        num_skip = int(data_directory[num_skipstr+4:num_uscore])
    else:
        num_skip = 0
    num_intervals = data_directory.find('_intervals')
    num_uscore_bf_intervals = data_directory[:num_intervals].rfind('_')
    time_interval = int(data_directory[num_uscore_bf_intervals+1:num_intervals-1])

    if 'frame' in data_directory:
        num_frame = data_directory.find('frame')
        num_uscore = num_frame + data_directory[num_frame:].find('_')
        # Photobleaching occurs prior to the activation cycle listed as frame## in the filename
        # Then there is indexing by 0 in python, thus subtracting by 2
        num_pb = (num_skip + 1) * (int(data_directory[num_frame+5:num_uscore]) - 1) - 1
    else:
        num_pb = np.nan
    df = pd.DataFrame([[num_skip, time_interval, num_pb]],
                    columns=['skip number','time interval (s)', 'photobleach frame number'])
    
    return df