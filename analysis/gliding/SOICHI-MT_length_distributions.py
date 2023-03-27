#%%
import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io, morphology, measure, filters
import os
import active_matter_pkg as amp
import arviz
import cmdstanpy
amp.viz.plotting_style()

pixel_size = 0.161 # µm per pixel

root1 = '../../../data/MT_length_distribution/2022-Sep1_MTdist_Box1_8'
root2 = '../../../data/MT_length_distribution/2022-Sep1_MTdist_Box2_9'
root3 = '../../../data/MT_length_distribution/2022-Sep1_MTdist_Box3_10'
root = [root1, root2, root3]
#tiff_list = active_matter_pkg.io.find_all_tiffs(root)

#%%
codepath = './MT_length_ model_code.stan'
MT_length_model = cmdstanpy.CmdStanModel(stan_file=codepath)
#%%
df_props = pd.DataFrame()
for _root in root:
    tiff_list = amp.io.find_all_tiffs(_root)
    for image_file in tiff_list:
        image = io.imread(image_file)
        im_processed = amp.image_processing.process_mt_images(image, block_size=5, 
                                                        mask_size=3, edge=5, 
                                                        area_thresh=10, min_dist=9,
                                                        min_angle=75, padding=3)
        im_props = measure.regionprops_table(im_processed, image,
                                        properties=['area', 'major_axis_length',
                                                    'minor_axis_length','centroid',
                                                    'orientation', 'eccentricity',
                                                    'label', 'perimeter'])
        _df = pd.DataFrame(im_props)
        _df['tiff_file'] = image_file
        # Remove non-existent objects from thinning to single-pixel size
        _df = _df[_df['perimeter']>0]
        df_props = df_props.append(_df, ignore_index=True)
#%%
#df_props.to_csv('../../analyzed_data/MT_lengths_sept2022.csv', sep=',')
#%%
mt_length = np.sort(df_props['perimeter'].values) * pixel_size
ecdf = np.arange(1, len(mt_length)+1, 1) /len(mt_length)

shortest = 0.01 #µm
longest = 100 #µm
dataset = dict(N = len(mt_length), l = mt_length,
                shortest=shortest, longest=longest)

samples = MT_length_model.sampling(data=dataset)
df_lambda = samples.to_dataframe()

x = mt_length.copy()
l_ = df_lambda['lambda'].values

y_cred = np.zeros((2,len(x)))
for i in range(len(x)):
        y_i = 1 - np.exp(- (x[i]) / l_)
        y_cred[:,i] = amp.stats.hpd(y_i,0.95)

lambda_med = np.median(df_lambda['lambda'].values)

x_theor = np.linspace(0, np.max(mt_length), 100)
mt_median = 1 - np.exp(- (x_theor) / lambda_med)

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.step(mt_length, ecdf, label='ECDF', color='dodgerblue', lw=2, zorder=4)
ax.fill_between(mt_length, y_cred[0,:], y_cred[1,:], 
                label='95% credible region', color='tomato', alpha=0.25)
ax.plot(x_theor, mt_median, color='tomato', zorder=3,
        label=r'$\lambda = %.1f$ µm' %lambda_med, lw=2)
ax.set_xlim([0.1,np .max(mt_length)])
ax.set_ylim([0.01,1.01])
ax.set_xlabel('microtubule length [µm]', fontsize=20)
ax.set_ylabel('cumulative distribution', fontsize=20)
ax.legend(fontsize=14, loc=4)
ax.set_xscale('linear')
#plt.savefig('../figures/MT_length_CDF_TR_data.pdf', bbox_inches='tight',
#            background_color='white')

#%%
fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.step(mt_length, ecdf, label='ECDF', color='dodgerblue', lw=2, zorder=4)
ax.fill_between(mt_length, y_cred[0,:], y_cred[1,:], 
                label='95% credible region', color='tomato', alpha=0.25)
ax.plot(x_theor, mt_median, color='tomato', zorder=3,
        label=r'$\lambda = %.1f$ µm' %lambda_med, lw=2)
ax.set_xlim([0.1,np.max(mt_length)])
ax.set_ylim([0.01,1.01])
ax.set_xlabel('microtubule length [µm]', fontsize=20)
ax.set_ylabel('cumulative distribution', fontsize=20)
ax.legend(fontsize=14, loc=4)
ax.set_xscale('log')
#plt.savefig('../../figures/MT_length_CDF_xlog_june2022_data.pdf', bbox_inches='tight',
#            background_color='white')

#%%
for _root in root:
    tiff_list = amp.io.find_all_tiffs(_root)
    for im in tiff_list:
        test_image = io.imread(im)
        im_filtered = amp.image_processing.filter_mts(test_image, block_size=5, mask_size=3)
        im_label, n_labels = measure.label(im_filtered, return_num=True)
        im_internal = amp.image_processing.border_clear(im_label, n_labels)
        im_sized = amp.image_processing.remove_small(im_internal, area_thresh=10)
        im_thinned = morphology.thin(im_sized)
        im_relabel = measure.label(im_thinned)
        im_noxovers = amp.image_processing.remove_line_crossovers(im_relabel, min_dist=9,
                                                            min_angle=75, padding=3)

        test_image[490:495,428:490] = test_image.max()

        fig, ax = plt.subplots(1,4,figsize=(16,4))
        ax[0].imshow(test_image)
        ax[0].set_title('raw image', fontsize=16)
        ax[0].text(0,-10, '(A)', fontsize=16, ha='left', va='bottom')

        ax[1].imshow(im_filtered)
        ax[1].set_title('binary image', fontsize=16)
        ax[1].text(0,-10, '(B)', fontsize=16, ha='left', va='bottom')

        ax[2].imshow(im_thinned)
        ax[2].set_title('edge and size excluded', fontsize=16)
        ax[2].text(0,-10, '(C)', fontsize=16, ha='left', va='bottom')

        ax[3].imshow(np.where(im_noxovers>1,1,im_noxovers))
        ax[3].set_title('processed images', fontsize=16)
        ax[3].text(0,-10, '(D)', fontsize=16, ha='left', va='bottom')
        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])
        fig.tight_layout()
#plt.savefig('../../figures/SIFigX_MT_image_processing_SHmanuscript.pdf', bbox_inches='tight',
#            background_color='white')
#%%
count_nums = np.zeros((2, len(tiff_list)))
count_nums[0,:] = np.arange(1, len(tiff_list)+1, 1)
for n in range(len(tiff_list)):
    test_image = io.imread(tiff_list[n])
    im_filtered = amp.image_processing.filter_mts(test_image, block_size=3, mask_size=3)
    im_label, n_labels = measure.label(im_filtered, return_num=True)
    im_sized = np.copy(im_label)

    unique, counts = np.unique(im_sized, return_counts=True)

    # Create dictionary except for 0 (background)
    count_nums[1,n] = unique[-1]

# %%
window = np.s_[200:300,250:350]
fig, ax = plt.subplots(2,2,figsize=(16,16))
ax[0,0].imshow(test_image[window])
ax[0,1].imshow(im_filtered[window])
ax[1,0].imshow(im_thinned[window])
ax[1,1].imshow(np.where(im_noxovers>1,1,im_noxovers)[window])
#plt.savefig('../figures/FigX_MT_image_processing_cropped_april2021_data.pdf', bbox_inches='tight',
#            background_color='white')
# %%
for image_file in tiff_list:
    area_thresh=10
    image = io.imread(image_file)
    im_filtered = amp.image_processing.filter_mts(image, block_size=5, mask_size=3)
    im_label, n_labels = measure.label(im_filtered, return_num=True)
    unique, _ = amp.image_processing.determine_count_nums(im_label)
    if unique[-1] > 200:
        im_filtered = amp.image_processing.filter_mts(image, block_size=5, mask_size=3, yen=True)
        im_label, n_labels = measure.label(im_filtered, return_num=True)
    im_internal = amp.image_processing.border_clear(im_label, edge=5)
    im_sized = amp.image_processing.remove_small(im_internal, area_thresh=10)
    im_thinned = morphology.thin(im_sized)
    im_relabel = measure.label(im_thinned)
    im_noxovers = amp.image_processing.remove_line_crossovers(im_relabel, min_dist=9,
                                                        min_angle=75, padding=3)
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    ax[0,0].imshow(image)
    ax[0,1].imshow(im_filtered)
    ax[1,0].imshow(im_thinned)
    ax[1,1].imshow(np.where(im_noxovers>1,1,im_noxovers))
    plt.show()

#%%
mt_length = np.sort(df_props['perimeter'].values) * pixel_size
ecdf = np.arange(1, len(mt_length)+1, 1) /len(mt_length)

shortest = 0.01 #µm
longest = 100 #µm

fig, ax = plt.subplots(1,1, figsize=(8,8))
ax.step(mt_length, ecdf, label='median = %.2f µm' %np.median(mt_length), color='dodgerblue', lw=2, zorder=4)
ax.set_xlim([shortest,longest])
ax.set_ylim([0.0,1.01])
ax.set_xlabel('microtubule length [µm]', fontsize=20)
ax.set_ylabel('cumulative distribution', fontsize=20)
ax.legend(fontsize=14, loc=4)
ax.set_xscale('log')
plt.savefig('../../figures/MT_lengthdist_june2022.pdf', bbox_inches='tight',
            facecolor='white')
# %%
test_image = io.imread(tiff_list[3])
im_filtered = image_processing.filter_mts(test_image, block_size=7, mask_size=7)
fig, ax = plt.subplots(2,1,figsize=(16,32))
ax[0].imshow(test_image)
ax[1].imshow(im_filtered)
# %%
