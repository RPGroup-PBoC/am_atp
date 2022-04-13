#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import active_matter_pkg as amp
amp.viz.plotting_style()

data_root = '../../data/atp/2021-06-10_Filter3_250ATP_mCherryiATP_1/'
imset = amp.io.find_all_tiffs(data_root)
# %%
files_bright = np.sort([imname for imname in imset if '/Bright/' in imname])
files_cherry = np.sort([imname for imname in imset if '/Cherry/' in imname])
files_iatp = np.sort([imname for imname in imset if '/iATP/' in imname])
# %%
im = io.imread(files_cherry[-2])
plt.figure(figsize=(16,8))
plt.imshow(im)
# %%
window = np.s_[500:700,900:1100]
plt.imshow(im[window])
# %%
plt.plot(im[window][125,:])
# %%
