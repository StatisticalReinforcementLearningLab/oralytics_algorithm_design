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
import experiment_global_vars

### GLOBAL VALUES ###
NUM_TRIAL_USERS = experiment_global_vars.NUM_TRIAL_USERS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES
MAX_SEED_VAL = experiment_global_vars.MAX_SEED_VAL
GRID_INCREMENT = 20
ALG_VALS = range(0, 190, GRID_INCREMENT)

READ_PATH_PREFIX = read_write_info.READ_PATH_PREFIX
WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX + "figures/"

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

def format_qualities(string_prefix):
  total_qualities = np.zeros(shape=(MAX_SEED_VAL, NUM_TRIAL_USERS, NUM_DECISION_TIMES))

  # extract pickle
  for i in range(MAX_SEED_VAL):
    try:
        pickle_name = string_prefix + "{}_data_df.p".format(i)
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

def create_grids(read_path_prefix):
    avg_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    low_perc_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    for i in ALG_VALS:
        for j in ALG_VALS:
            string_prefix = read_path_prefix + "_[{}, {}]/".format(i, j)
            print("For: ", string_prefix)
            total_rewards = format_qualities(string_prefix)
            avg_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_mean_quality(total_rewards)
            low_perc_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_lower_25_quality(total_rewards)

    return avg_grid, low_perc_grid

def create_and_save_plots(exp_path):
    avg_grid, low_25_grid = create_grids(exp_path)
    sim_env = exp_path.split(READ_PATH_PREFIX)[1].split('/')[1]
    print("SIM ENV NAME!", sim_env)
    with open(WRITE_PATH_PREFIX + "{}_AVG_HEATMAP.p".format(sim_env), 'wb') as f:
        pickle.dump(avg_grid, f)
    with open(WRITE_PATH_PREFIX + "{}_25_PERC_HEATMAP.p".format(sim_env), 'wb') as f:
        pickle.dump(low_25_grid, f)
