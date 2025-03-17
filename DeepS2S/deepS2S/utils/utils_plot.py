import os
from math import log10, ceil, floor 
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


def set_fig_size(width='thesis', width_fraction=1, subplots=(1, 1), subplot_fraction=1, ratio=None):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 399.47578
    else:
        width_pt = width

    #Width of figure (in pts)
    fig_width_pt = width_pt * width_fraction   
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    if ratio is None:
        # golden ratio
        ratio = (5**.5 - 1) / 2
    
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = (fig_width_in * ratio * (subplots[0] / subplots[1])) / subplot_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def label_axis(ax, values, mpl_ax, v_min=None, v_max=None):
    v = list(values)
    
    if v_min is None:
        v_min = np.min(v)
    if v_max is None:
        v_max = np.max(v)

    exp = -int(floor(log10(abs(v_max))))

    bottom = round(v_min, exp) 
    top = round(v_max, exp)

    if bottom > v_min:
        bottom  -= 10**(-exp)
    
    if top < v_max:
        top += 10**(-exp)
    
    if ax == 'x':
        mpl_ax.set_xlim(v_min, v_max)
        mpl_ax.set_xticks(np.linspace(bottom, top, 5))
    elif ax == 'y':
        mpl_ax.set_ylim(v_min, v_max)
        mpl_ax.set_yticks(np.linspace(bottom, top, 5))



def label_axes_grid(fig, axes, xlims=None, ylims=None, xlabel=None, ylabels=None, ylabel=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, grid_x=True, grid_y=True):
    for row_i, row_axes in enumerate(axes):
        for col_i, ax in enumerate(row_axes):
            if xticks is not None: ax.set_xticks(xticks)
            if yticks is not None: ax.set_yticks(yticks)
            if xlims is not None: ax.set_xlim(xlims)
            if ylims is not None: ax.set_ylim(ylims)
            if xticklabels is not None: ax.set_xticklabels(xticklabels)
            if yticklabels is not None: ax.set_yticklabels(yticklabels)
            if xlabel is not None:
                if row_i == len(axes)-1:
                    if isinstance(xlabel, list):
                        ax.set_xlabel(xlabel[col_i])
            if ylabels is not None:
                if col_i == 0:
                    if isinstance(ylabels, list):
                        ax.set_ylabel(ylabels[row_i])

            ax.grid(visible=False, axis='both')
            if grid_x: ax.grid(visible=True, axis='x')
            if grid_y: ax.grid(visible=True, axis='y')

            if row_i < len(axes)-1:
                ax.set_xticklabels([])
            if col_i != 0:
                ax.set_yticklabels([])

    if isinstance(xlabel, str):
        fig.text(0.5, 0.05, xlabel, ha='center')
    if isinstance(ylabel, str):
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

def plot_class_dist(preds,
                    targets,
                    xticklabs = ['NAO+', 'SB', 'AR', 'NAO-']):
    """Plot distribution of classes for each leadtime comparing predicted vs target labels

    Args:
        preds (np.ndarray): predicted class labels
        targets (np.ndarray): traget class labels
        xticklabs (list): list of class names

    Returns:
        fig-object: figure object
    """
    fig, axes = plt.subplots(6,2, figsize=set_fig_size(subplots=(6,2)))
    y_max = 0
    for row in range(6):
        # how often was each class predicted in this time step
        counts = [sum(i == preds[:, row]) for i in range(4)]
        y_max = max(counts) if max(counts) > y_max else y_max
        axes[row][0].bar(x=np.arange(4), height=counts)
        axes[row][0].set_title(f'Predicted: Leadtime = week {row+1}')
        # how often does each class occur in this time step
        counts = [sum(i == targets[:, row]) for i in range(4)]
        y_max = max(counts) if max(counts) > y_max else y_max
        axes[row][1].bar(x=np.arange(4), height=counts)
        axes[row][1].set_title(f'Actual: Leadtime = week {row+1}')

    label_axes_grid(fig, axes, ylims=(0, y_max+10), xticks=np.arange(4), xticklabels=xticklabs, xlabel='Classes', ylabel='Number of samples', yticks=np.arange(0, y_max+10, 50), grid_x=False)


    plt.show()
    return fig

def plot_joint_class_dist(pred_dict,
                    targets,
                    cm_list,
                    xticklabs = ['NAO+', 'SB', 'AR', 'NAO-']):
    """Plot distribution of classes for each leadtime comparing predicted vs target labels

    Args:
        preds (np.ndarray): predicted class labels
        targets (list): traget class labels
        xticklabs (list): list of class names

    Returns:
        fig-object: figure object
    """
    # fig, axes = plt.subplots(6,len(pred_dict)+1, figsize=plot_utils.set_fig_size(subplots=(6,len(pred_dict)+1)), sharey = True)
    fig, axes = plt.subplots(6,len(pred_dict), figsize=(20,15))#, sharey = True)
    y_max = 0
    col  = 0
    for keys, preds in pred_dict.items():
        # preds = pred_dict[keys]
        print(preds.shape)
        for row in range(6):
            # how often does each class occur in this time step
            counts_tar = [sum(i == targets[:, row]) for i in range(4)]
            y_max = max(counts_tar) if max(counts_tar) > y_max else y_max
            axes[row][col].bar(x=np.arange(4), height=counts_tar, alpha = 0.5, color ='#000000', label = 'target')
            # how often was each class predicted in this time step
            counts = [sum(i == preds[:, row]) for i in range(4)]
            y_max = max(counts) if max(counts) > y_max else y_max
            axes[row][col].bar(x=np.arange(4), height=counts, alpha = 0.5, color =cm_list[col], label = 'predictions')
            # axes[row][col].legend()
            if col == 0:
                # counts_tar = [sum(i == targets[:, row]) for i in range(4)]
                # y_max = max(counts_tar) if max(counts_tar) > y_max else y_max
                # axes[row][-1].bar(x=np.arange(4), height=counts_tar)
                axes[row][col].set_ylabel(f'Leadtime= week {row+1}')
            if row == 0:
                axes[row][col].set_title(f'{keys} predictions')
                # axes[row][-1].set_title(f'Target distribution')
        col +=1 

    label_axes_grid(fig, axes, ylims=(0, y_max+10), xticks=np.arange(4), xticklabels=xticklabs, xlabel='Classes', ylabel='Number of samples', yticks=np.arange(0, y_max+10, 50), grid_x=False)


    plt.show()
    return fig

def plot_examples(ds, examples=None):
    if isinstance(ds, xr.Dataset):
        ds = ds.to_array().squeeze()
    if examples is None:
        examples = np.random.choice(np.arange(len(ds.time)), 12)
    
    extent = (ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max())

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(4, 3,figsize=(20, 20), subplot_kw={'projection': proj})
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.coastlines()
        p = ax.imshow(ds.isel(time=examples[i]), extent=extent, origin='lower', cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
        fig.colorbar(p, ax=ax, location='bottom')
