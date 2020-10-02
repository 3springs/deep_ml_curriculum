import numpy as np
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

color_dict = {'Aeolian Sandstone': '#ffffe0',
 'Anhydrite': '#ff80ff',
 'Argillaceous Limestone': '#1e90ff',
 'Arkose': '#eedd82',
 'Basement': '#fa8072',
 'Biogenic Ooze': '#CCCC00',
 'Calcareous Cement': '#00ffff',
 'Calcareous Debris Flow': '#40e0d0',
 'Calcareous Shale': '#008b8b',
 'Carnallite': '#ff00ff',
 'Chalk': '#6a5acd',
 'Cinerite': '#00ffff',
 'Coal': '#000000',
 'Conglomerate': '#ffffe0',
 'Cross Bedded Sst': '#ffd700',
 'Dolomite': '#00ffff',
 'Gap': '#ffffff',
 'Halite': '#ffc0cb',
 'Kaïnite': '#fff0f5',
 'Limestone': '#6a5acd',
 'Marlstone': '#00bfff',
 'Metamorphic Rock': '#008b8b',
 'Plutonic Rock': '#ff0000',
 'Polyhalite': '#ffb6c1',
 'Porous Limestone': '#6a5acd',
 'Sandstone': '#ffff00',
 'Sandy Silt': '#d2b48c',
 'Shale': '#008b8b',
 'Shaly Silt': '#CCCC00',
 'Silt': '#ffa07a',
 'Silty Sand': '#ffffe0',
 'Silty Shale': '#006400',
 'Spiculite': '#939799',
 'Sylvinite': '#ff80ff',
 'Volcanic Rock': '#ffa500',
 'Volcanic Tuff': '#ff6347',
}

def remove_unused_categories(cats):
    """Takes in list of pd.Categorical. Gives them same, cleaned categories."""
    new_categories = []
    for c in cats:
        new_categories += list(set(c))
    new_categories = sorted(set(new_categories))
    
    cats = [pd.Categorical(c, categories=new_categories) for c in cats]
    return cats

def plot_facies(facies:pd.Categorical, ax=None, colorbar=True, xlabel='Facies'):
    """Plot as facies log.
    
    Facies must be a pandas categorical series e.g. pd.Categorical(['Sst', 'Lmst', 'SSt'])
    """
    if ax is None:
        ax = plt.gca()
    facies_colors = [color_dict.get(f, 'white') for f in facies.categories]
    
    # Plot facies as image
    cluster=np.repeat(np.expand_dims(facies.codes,1), 100, 1)
    
    # custom qualitative colormap
    cmap_facies = colors.ListedColormap(
            facies_colors[0:len(facies_colors)], 'indexed')
    
    # the 0.5 is to center the labels
    im=ax.imshow(cluster, interpolation='none', aspect='auto',
                        cmap=cmap_facies,vmin=-0.5,vmax=len(facies.categories)-0.5)
    ax.set_xlabel(xlabel)

    divider = make_axes_locatable(ax)
    if colorbar:
        cax = divider.append_axes("right", size="20%", pad=0.05)

        # modified from https://gist.github.com/jakevdp/8a992f606899ac24b711
        # This function formatter will replace integers with target names
        formatter = plt.FuncFormatter(lambda val, loc: facies.categories[val])

        # We must be sure to specify the ticks matching our target names
        plt.colorbar(im, ticks=range(len(facies.categories)), format=formatter, cax=cax)
        ax.set_xticklabels([])


def plot_well(well_name:str, logs:pd.DataFrame, facies:pd.Categorical, figsize=(8, 12)):   

    ztop=logs.DEPT.min(); zbot=logs.DEPT.max()
    f, ax = plt.subplots(nrows=1, ncols=5, figsize=(8, 12))

    ax[0].plot(logs.GR, logs.DEPT, '-g')
    ax[0].set_xlabel("GR")

    ax[1].plot(logs.CALI, logs.DEPT, '-')
    ax[1].set_xlabel("CALI")

    ax[2].plot(logs.RDEP, logs.DEPT, '-r', alpha=0.7)
    ax[2].plot(logs.RMED, logs.DEPT, '-g', alpha=0.7)
    ax[2].set_xlim(logs.RDEP.min(),100)
    ax[2].set_xlabel("RDEP (r) & RMED (g)")

    ax[3].plot(logs.RHOB, logs.DEPT, '-')
    ax[3].set_xlabel("RHOB")

    [facies] = remove_unused_categories([facies])
    plot_facies(facies, ax[-1])
    
    for i in range(len(ax)-1):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)

    ax[1].set_yticklabels([]); ax[2].set_yticklabels([]); ax[3].set_yticklabels([])
    ax[4].set_yticklabels([]); 
    ax[-1].set_xticklabels([])
    f.suptitle('Well: %s'%well_name, fontsize=14,y=0.94)
    return f, ax
    

def plot_well_pred(well_name:str, logs:pd.DataFrame, facies_true:pd.Categorical, facies_pred=pd.Categorical, figsize=(8, 12)):   

    ztop=logs.DEPT.min(); zbot=logs.DEPT.max()
    f, ax = plt.subplots(nrows=1, ncols=6, figsize=figsize)

    ax[0].plot(logs.GR, logs.DEPT, '-g')
    ax[0].set_xlabel("GR")

    ax[1].plot(logs.CALI, logs.DEPT, '-')
    ax[1].set_xlabel("CALI")

    ax[2].plot(logs.RDEP, logs.DEPT, '-r', alpha=0.7)
    ax[2].plot(logs.RMED, logs.DEPT, '-g', alpha=0.7)
    ax[2].set_xlim(logs.RDEP.min(),100)
    ax[2].set_xlabel("RDEP (r) & RMED (g)")

    ax[3].plot(logs.RHOB, logs.DEPT, '-')
    ax[3].set_xlabel("RHOB")

    [facies_pred, facies_true] = remove_unused_categories([facies_pred, facies_true])
    assert (facies_pred.categories == facies_true.categories).all()
    plot_facies(facies_pred, ax=ax[4], colorbar=False, xlabel='Facies (pred)')
    plot_facies(facies_true, ax=ax[5], xlabel='Facies (true)')
    
    for i in range(len(ax)-2):
        ax[i].set_ylim(ztop,zbot)
        ax[i].invert_yaxis()
        ax[i].grid()
        ax[i].locator_params(axis='x', nbins=3)


    for i in range(1, len(ax)):
        ax[i].set_yticklabels([])

    ax[-2].set_xticklabels([])
    ax[-1].set_xticklabels([])
    f.suptitle('Well: %s'%well_name, fontsize=14,y=0.94)
    return f, ax
