# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set_theme()
import itertools

import simulation_environment
import rl_algorithm
import read_write_info

### GLOBAL VALUES ###
NUM_TRIAL_USERS = 72
NUM_DECISION_POINTS = 140
NUM_TRIALS = 100
GRID_INCREMENT = 20

ENV_VARIANTS = dict(
    BASE_NAMES = ["STAT_LOW_R", "STAT_MED_R", "STAT_HIGH_R",\
     "NON_STAT_LOW_R", "NON_STAT_MED_R", "NON_STAT_HIGH_R"],
    EFFECT_SIZE_SCALE=['small', 'smaller']
)

ENV_NAMES = ["{}_{}".format(*sim_env_params) for sim_env_params in itertools.product(*list(ENV_VARIANTS.values()))]
ALG_VALS = range(0, 190, GRID_INCREMENT)

READ_PATH_PREFIX = read_write_info.READ_PATH_PREFIX + "hyper_pickle_results/"
WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX + "figures/"

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

def format_qualities(string_prefix):
  total_qualities = np.zeros(shape=(NUM_TRIALS, NUM_TRIAL_USERS, NUM_DECISION_POINTS))

  # extract pickle
  for i in range(NUM_TRIALS):
    try:
        pickle_name = string_prefix + "_{}_data_df.p".format(i)
        data_df = pd.read_pickle(pickle_name)
    except:
        print("Couldn't for {}".format(pickle_name))
    # user_idx is both the unique identifier and an index
    for user_idx in np.unique(data_df['user_idx']):
        user_qualities = get_data_df_values_for_users(data_df, [user_idx], 'quality').flatten()
        total_qualities[i][user_idx] = user_qualities

  return total_qualities

 # average across trajectories, average across users,
#  average across trials
def report_mean_quality(total_rewards):
  a = np.mean(np.mean(total_rewards, axis=2), axis=1)

  return np.mean(a), stats.sem(a)

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_quality(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)

  return np.mean(a), stats.sem(a)

def report_upper_75_quality(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 75, axis=1)

  return np.mean(a), stats.sem(a)

def create_grids(env_name):
    avg_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    low_perc_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    # high_perc_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    for i in ALG_VALS:
        for j in ALG_VALS:
            string_prefix = READ_PATH_PREFIX + "{}_{}_{}".format(env_name, i, j)
            print("For: ", string_prefix)
            total_rewards = format_qualities(string_prefix)
            avg_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_mean_quality(total_rewards)
            low_perc_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_lower_25_quality(total_rewards)
            # high_perc_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_upper_75_quality(total_rewards)


    return avg_grid, low_perc_grid

for SIM_ENV in ENV_NAMES:
    avg_grid, low_25_grid = create_grids(SIM_ENV)
    with open(WRITE_PATH_PREFIX + "{}_AVG_HEATMAP.p".format(SIM_ENV), 'wb') as f:
        pickle.dump(avg_grid, f)
    with open(WRITE_PATH_PREFIX + "{}_25_PERC_HEATMAP.p".format(SIM_ENV), 'wb') as f:
        pickle.dump(low_25_grid, f)
    # with open(WRITE_PATH_PREFIX + "{}_75_PERC_HEATMAP.p".format(SIM_ENV), 'wb') as f:
    #     pickle.dump(high_75_grid, f)
