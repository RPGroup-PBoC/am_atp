"""
viz.py

Module for calling standard plotting functions and setting the styles.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from bokeh.io import curdoc
from bokeh.themes import Theme

def plotting_style():
    """
    Sets matplotlibs plotting style to the publication default. It returns a
    list of the preferred colors.
    """
    rc = {'axes.facecolor': '#e7d8b8',
          'axes.grid': False,
          'axes.frameon': True,
          'ytick.direction': 'out',
          'xtick.direction': 'out',
          'xtick.gridOn': True,
          'ytick.gridOn': True,
          'ytick.major.width':5,
          'xtick.major.width':5,
          'ytick.major.size': 5,
          'xtick.major.size': 5,
          'mathtext.fontset': 'stixsans',
          'mathtext.sf': 'sans',
          'legend.frameon': True,
          'legend.facecolor': '#FFEDCE',
          'figure.dpi': 150,
          'xtick.color': 'k',
          'ytick.color': 'k'}
    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('dark', rc=rc)
    mpl.rcParams['image.cmap'] = 'gray'
    mpl.rcParams['xtick.labelsize']=16
    mpl.rcParams['ytick.labelsize']=16
    mpl.rcParams['axes.labelsize']=20
    mpl.rcParams['figure.titlesize']=20

def plotting_style_bokeh():
    """
    Sets bokeh plotting style to the publication default. It returns a list
    of the preferred colors.
    """
    curdoc().theme = Theme(json={'attrs': {

        # apply defaults to Figure properties
        'Figure': {
                'toolbar_location': None,
                'outline_line_color': None,
                'min_border_right': 10,
                'background_fill_color': '#e7d8b8',
        },

        # apply defaults to Axis properties
        'Axis': {
                'major_tick_in': None,
                'major_tick_out': 3,
                'minor_tick_in': None,
                'minor_tick_out': None,
                'axis_line_color': None,
                'major_tick_line_color': '#000000',
                'axis_label_text_font_size': "14pt",
                'major_label_text_font_size': "12pt",
        },

        # apply properties to Grid
        'Grid': {
                'visible': False,
                'grid_line_color': '#FFFFFF',
                'grid_line_alpha': 0.75
        },

        # apply defaults to Legend properties
        'Legend': {
                'background_fill_alpha': 0,
                'background_fill_color': '#FFEDCE'
        }
    }})

def imshow_four(image1, image2, image3, image4, window=None):
    fig, ax = plt.subplots(2,2,figsize=(20,20))
    if window==None:
        ax[0,0].imshow(image1)
        ax[0,1].imshow(image2)
        ax[1,0].imshow(image3)
        ax[1,1].imshow(image4)
    else:
        ax[0,0].imshow(image1[window])
        ax[0,1].imshow(image2[window])
        ax[1,0].imshow(image3[window])
        ax[1,1].imshow(image4[window])
    plt.show()
    return