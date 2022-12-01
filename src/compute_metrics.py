import pandas as pd
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import itertools

### GLOBAL VALUES ###
NUM_TRIAL_USERS = 72
NUM_DECISION_POINTS = 140
NUM_TRIALS = 100

ENV_NAMES = ["STAT_LOW_R", "STAT_MED_R", "STAT_HIGH_R",\
 "NON_STAT_LOW_R", "NON_STAT_MED_R", "NON_STAT_HIGH_R"]

ALG_CANDIDATES = dict(
    # ALG_TYPES=["BLR_AC", "BLR_NO_AC"],
    ALG_TYPES=["BLR_AC"],
    B_LOGISTICS=[0.515, 5.15],
    CLIPPING_VALS=["0.2_0.8"],
    UPDATE_CADENCE=[14, 2],
    CLUSTER_SIZE=[72, 1],
    EFFECT_SIZE_SCALE=['small', 'smaller']
)

# ALG_NAMES = ["{}_{}".format(alg_type, b) for alg_type in ALG_TYPES for b in B_LOGISTICS]
ALG_NAMES = ["{}_{}_{}_{}_{}_{}".format(*candidate_params) for candidate_params in itertools.product(*list(ALG_CANDIDATES.values()))]
print(ALG_NAMES)


READ_PATH_PREFIX = "pickle_results/"
WRITE_PATH_PREFIX = "figures/"

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

# string_prefix is env type, clipping vals, b logistic val
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

  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_quality(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)

  return "{:.3f} ({:.3f})".format(round(np.mean(a), 3), round(stats.sem(a), 3))

### x axis is simulation environments ###
all_avg_qualities = []
all_lower_25_qualities = []

for SIM_ENV in ENV_NAMES:
    sim_env_avg_qualities = []
    sim_env_lower_25_qualities = []
    for ALG_NAME in ALG_NAMES:
        print("FOR {} {}".format(ALG_NAME, SIM_ENV))
        string_prefix = READ_PATH_PREFIX + "{}_{}".format(SIM_ENV, ALG_NAME)
        alg_qualities = format_qualities(string_prefix)
        sim_env_avg_qualities.append(report_mean_quality(alg_qualities))
        sim_env_lower_25_qualities.append(report_lower_25_quality(alg_qualities))

    all_avg_qualities.append(sim_env_avg_qualities)
    all_lower_25_qualities.append(sim_env_lower_25_qualities)

# formatting metrics into df and then convert to latex
total_avg_vals = dict(ALG_CANDS=ALG_NAMES)
avg_vals = {ENV_NAMES[i]: all_avg_qualities[i] for i in range(len(ENV_NAMES))}
total_avg_vals.update(avg_vals)
df_avg_qualities = pd.DataFrame(total_avg_vals)

total_lower_25_vals = dict(ALG_CANDS=ALG_NAMES)
lower_25_vals = {ENV_NAMES[i]: all_lower_25_qualities[i] for i in range(len(ENV_NAMES))}
total_lower_25_vals.update(lower_25_vals)
df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

### x axis is algorithm candidates ###
# all_avg_qualities = []
# all_lower_25_qualities = []
#
# for ALG_NAME in ALG_NAMES:
#     alg_avg_qualities = []
#     alg_lower_25_qualities = []
#     for SIM_ENV in ENV_NAMES:
#         print("FOR {} {}".format(ALG_NAME, SIM_ENV))
#         string_prefix = READ_PATH_PREFIX + "{}_{}".format(SIM_ENV, ALG_NAME)
#         alg_qualities = format_qualities(string_prefix)
#         alg_avg_qualities.append(report_mean_quality(alg_qualities))
#         alg_lower_25_qualities.append(report_lower_25_quality(alg_qualities))
#
#     all_avg_qualities.append(alg_avg_qualities)
#     all_lower_25_qualities.append(alg_lower_25_qualities)

# formatting metrics into df and then convert to latex
# total_avg_vals = dict(SIM_ENV=ENV_NAMES)
# avg_vals = {ALG_NAMES[i]: all_avg_qualities[i] for i in range(len(ALG_NAMES))}
# total_avg_vals.update(avg_vals)
# df_avg_qualities = pd.DataFrame(total_avg_vals)
#
# total_lower_25_vals = dict(SIM_ENV=ENV_NAMES)
# lower_25_vals = {ALG_NAMES[i]: all_lower_25_qualities[i] for i in range(len(ALG_NAMES))}
# total_lower_25_vals.update(lower_25_vals)
# df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

print(df_avg_qualities.to_latex(index=False))
print(df_lower_25_qualities.to_latex(index=False))
