import pandas as pd
import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import read_write_info
import run

### GLOBAL VALUES ###
READ_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX

NUM_TRIAL_USERS = 72
NUM_DECISION_POINTS = 140
NUM_TRIALS = 100

# gets the current experiment name from queue
EXPERIMENT_DIR = run.QUEUE[0][0]

# ENV_NAMES = ["{}_{}".format(env_base, eff_size) for env_base in run.SIM_ENV_TYPES for eff_size in run.EFFECT_SIZE_SCALES]
ENV_NAMES_SMALL = ["{}_{}".format(env_base, "small") for env_base in run.SIM_ENV_TYPES]
ENV_NAMES_SMALLER = ["{}_{}".format(env_base, "smaller") for env_base in run.SIM_ENV_TYPES]

ALG_CANDIDATES = dict(
    ALG_TYPES=["BLR_AC", "BLR_NO_AC"],
    B_LOGISTICS=[0.515, 5.15],
    CLIPPING_VALS=[[0.2, 0.8]],
    UPDATE_CADENCE=[14, 2],
    CLUSTER_SIZE=[72, 1],
    NOISE_VARS=[3396.449, 3412.422]
)

EXP_PATH = READ_PATH_PREFIX + EXPERIMENT_DIR + "/"
# TODO: fill in algorithm candidate dimensions that were used in the experiment
ALG_NAMES = ["{}".format(noise_var) for noise_var in ALG_CANDIDATES["NOISE_VARS"]]
# ALG_NAMES = ["{}_{}_{}_{}_{}_{}".format(*candidate_params) for candidate_params in itertools.product(*list(ALG_CANDIDATES.values()))]
print("ALG_NAMES", ALG_NAMES)

def get_data_df_values_for_users(data_df, user_idxs, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs))].filter(regex=(regex_pattern)))

# string_prefix is env type, clipping vals, b logistic val
def format_qualities(folder_name):
  total_qualities = np.zeros(shape=(NUM_TRIALS, NUM_TRIAL_USERS, NUM_DECISION_POINTS))

  # extract pickle
  for i in range(NUM_TRIALS):
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

### x axis is simulation environments ###
all_avg_qualities = []
all_lower_25_qualities = []

# for SIM_ENV in ENV_NAMES:
for SIM_ENV in ENV_NAMES_SMALLER:
    sim_env_avg_qualities = []
    sim_env_lower_25_qualities = []
    for ALG_NAME in ALG_NAMES:
        print("FOR {} {}".format(ALG_NAME, SIM_ENV))
        folder_name = EXP_PATH + "{}_{}".format(SIM_ENV, ALG_NAME)
        alg_qualities = format_qualities(folder_name)
        sim_env_avg_qualities.append(report_mean_quality(alg_qualities))
        sim_env_lower_25_qualities.append(report_lower_25_quality(alg_qualities))

    all_avg_qualities.append(sim_env_avg_qualities)
    all_lower_25_qualities.append(sim_env_lower_25_qualities)

# formatting metrics into df and then convert to latex
total_avg_vals = dict(ALG_CANDS=ALG_NAMES)
avg_vals = {ENV_NAMES_SMALLER[i]: all_avg_qualities[i] for i in range(len(ENV_NAMES_SMALLER))}
total_avg_vals.update(avg_vals)
df_avg_qualities = pd.DataFrame(total_avg_vals)

total_lower_25_vals = dict(ALG_CANDS=ALG_NAMES)
lower_25_vals = {ENV_NAMES_SMALLER[i]: all_lower_25_qualities[i] for i in range(len(ENV_NAMES_SMALLER))}
total_lower_25_vals.update(lower_25_vals)
df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

print(df_avg_qualities.to_latex(index=False))
print(df_lower_25_qualities.to_latex(index=False))

# for SIM_ENV in ENV_NAMES:
#     sim_env_avg_qualities = []
#     sim_env_lower_25_qualities = []
#     for ALG_NAME in ALG_NAMES:
#         print("FOR {} {}".format(ALG_NAME, SIM_ENV))
#         folder_name = EXP_PATH + "{}_{}".format(SIM_ENV, ALG_NAME)
#         alg_qualities = format_qualities(folder_name)
#         sim_env_avg_qualities.append(report_mean_quality(alg_qualities))
#         sim_env_lower_25_qualities.append(report_lower_25_quality(alg_qualities))
#
#     all_avg_qualities.append(sim_env_avg_qualities)
#     all_lower_25_qualities.append(sim_env_lower_25_qualities)
#
# # formatting metrics into df and then convert to latex
# total_avg_vals = dict(ALG_CANDS=ALG_NAMES)
# avg_vals = {ENV_NAMES[i]: all_avg_qualities[i] for i in range(len(ENV_NAMES))}
# total_avg_vals.update(avg_vals)
# df_avg_qualities = pd.DataFrame(total_avg_vals)
#
# total_lower_25_vals = dict(ALG_CANDS=ALG_NAMES)
# lower_25_vals = {ENV_NAMES[i]: all_lower_25_qualities[i] for i in range(len(ENV_NAMES))}
# total_lower_25_vals.update(lower_25_vals)
# df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

all_avg_qualities = []
all_lower_25_qualities = []

for SIM_ENV in ENV_NAMES_SMALL:
    sim_env_avg_qualities = []
    sim_env_lower_25_qualities = []
    for ALG_NAME in ALG_NAMES:
        print("FOR {} {}".format(ALG_NAME, SIM_ENV))
        folder_name = EXP_PATH + "{}_{}".format(SIM_ENV, ALG_NAME)
        alg_qualities = format_qualities(folder_name)
        sim_env_avg_qualities.append(report_mean_quality(alg_qualities))
        sim_env_lower_25_qualities.append(report_lower_25_quality(alg_qualities))

    all_avg_qualities.append(sim_env_avg_qualities)
    all_lower_25_qualities.append(sim_env_lower_25_qualities)

# formatting metrics into df and then convert to latex
total_avg_vals = dict(ALG_CANDS=ALG_NAMES)
avg_vals = {ENV_NAMES_SMALL[i]: all_avg_qualities[i] for i in range(len(ENV_NAMES_SMALL))}
total_avg_vals.update(avg_vals)
df_avg_qualities = pd.DataFrame(total_avg_vals)

total_lower_25_vals = dict(ALG_CANDS=ALG_NAMES)
lower_25_vals = {ENV_NAMES_SMALL[i]: all_lower_25_qualities[i] for i in range(len(ENV_NAMES_SMALL))}
total_lower_25_vals.update(lower_25_vals)
df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

print(df_avg_qualities.to_latex(index=False))
print(df_lower_25_qualities.to_latex(index=False))

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
