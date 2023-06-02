import pandas as pd
import numpy as np
import pickle
from scipy import stats
import itertools
import run
import experiment_global_vars

### GLOBAL VALUES ###
NUM_TRIAL_USERS = experiment_global_vars.NUM_TRIAL_USERS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

# string_prefix is env type, clipping vals, b logistic val
def format_qualities(folder_name, max_seed_val):
  total_qualities = np.zeros(shape=(max_seed_val, NUM_TRIAL_USERS, NUM_DECISION_TIMES))

  # extract pickle
  for i in range(max_seed_val):
    try:
        pickle_name = folder_name + "/{}_data_df.p".format(i)
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

  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_quality(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)

  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

def get_metric_values(folder_name, max_seed_val):
    total_qualities = format_qualities(folder_name, max_seed_val)

    return report_mean_quality(total_qualities), report_lower_25_quality(total_qualities)
