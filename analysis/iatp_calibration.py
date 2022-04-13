#%%
# Image analysis for iATP calibration. Saves extracted information into analyzed_data folder
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, filters, measure
import active_matter_pkg as amp
amp.viz.plotting_style()

data_root = '../../data/atp/iatp_calibration/2021-06-10_Filter3_ATPCal_200LED_1'

imset = amp.io.find_all_tiffs(data_root)
imset = np.sort(imset)
# %%
imset[0]
# %%
