# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pickle
import itertools
import re
from matplotlib.patches import Rectangle
import read_write_info

### GLOBAL VALUES ###
ENV_VARIANTS = dict(
    BASE_NAMES = ["STAT_LOW_R", "STAT_MED_R", "STAT_HIGH_R",\
     "NON_STAT_LOW_R", "NON_STAT_MED_R", "NON_STAT_HIGH_R"],
    EFFECT_SIZE_SCALE=['small', 'smaller', 'None']
)

ENV_NAMES = ["{}_{}".format(*sim_env_params) for sim_env_params in itertools.product(*list(ENV_VARIANTS.values()))]
E_MAPPING = {}
for env_name in ENV_NAMES:
    if re.search("LOW_R", env_name):
        E_MAPPING[env_name] = 0
    elif re.search("MED_R", env_name):
        E_MAPPING[env_name] = 0.5
    elif re.search("HIGH_R", env_name):
        E_MAPPING[env_name] = 0.8
    else:
        print("Error: none of the above")

READ_PATH_PREFIX = read_write_info.READ_PATH_PREFIX + "figures/"
WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX + "figures/"

# sns colors: https://seaborn.pydata.org/tutorial/color_palettes.html
def plot_heatmap(grid, color_scheme, title_val, rectangle_color, file_name=None, save_fig=False):
    # clear subplots
    plt.figure()
    sns.set(font_scale=1.2)
    ax = sns.heatmap(grid,cmap=sns.color_palette(color_scheme, as_cmap=True), \
                     xticklabels=1, yticklabels=1)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\xi_1$', fontsize=20)
    ax.set_ylabel(r'$\xi_2$', fontsize=20)
    ax.set_xticklabels(np.arange(0, 181, 20))
    ax.set_yticklabels(np.arange(0, 181, 20))
    ax.set_title(r'$E={}$'.format(title_val), fontsize=20)

    yloc, xloc = np.where(grid == np.amax(grid))
    ax.add_patch(Rectangle((xloc, yloc),1,1, fill=False, edgecolor=rectangle_color, lw=3))

    if save_fig:
        fig = ax.get_figure()
        fig.savefig(WRITE_PATH_PREFIX + "{}.pdf".format(file_name), bbox_inches="tight")

for SIM_ENV in ENV_NAMES:
    print("For Env: ", SIM_ENV)
    with open(READ_PATH_PREFIX + '{}_AVG_HEATMAP.p'.format(SIM_ENV), 'rb') as f:
        avg_grid = pickle.load(f)

    with open(READ_PATH_PREFIX + '{}_25_PERC_HEATMAP.p'.format(SIM_ENV), 'rb') as f:
        low_25_grid = pickle.load(f)

    title_val = E_MAPPING[SIM_ENV]

    plot_heatmap(avg_grid.T, "ch:start=.2,rot=-.3", title_val, 'cyan', "avg_heatmap_{}".format(SIM_ENV),  True)
    plot_heatmap(low_25_grid.T, "ch:s=-.2,r=.6", title_val, 'red', "25_perc_heatmap_{}".format(SIM_ENV),  True)
