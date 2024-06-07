# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

DATA = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
# get all robas 3 users
STUDY_USERS = np.unique(DATA['robas id'])
COLS = ['robas id', 'timeOfDay', 'brushingDuration', 'pressureDuration']
ALG_FEATURES_DF = DATA[COLS]
# num. of baseline featuers we have data on
D = 3

# grab user specific df
def get_user_df(user_id, df):
  return df[df['robas id'] == user_id]

GAMMA = 13/14
DISCOUNTED_GAMMA_ARRAY = GAMMA ** np.flip(np.arange(14))
CONSTANT = (1 - GAMMA) / (1 - GAMMA**14)

# brushing duration is of length 14 where the first element is the brushing duration
# at time t - 14 and the last element the brushing duration at time t - 1
def calculate_b_bar(brushing_durations):
    sum_term = DISCOUNTED_GAMMA_ARRAY * brushing_durations

    return CONSTANT * np.sum(sum_term)

# b bar is designed to be in [0, 180]
def normalize_b_bar(b_bar):
    return (b_bar - (181 / 2)) / (179 / 2)

# now impute the correct state_b_bar and state_app_engage
# instead of the one used for action selection
def get_normalize_b_bar(user_qualities):
    j = len(user_qualities)
    if j < 14:
        b_bar = 0 if j < 1 else np.mean(user_qualities)
    else:
        b_bar = calculate_b_bar(user_qualities[-14:])

    return normalize_b_bar(b_bar)

def get_quality_array(data_df, user_decision_t):
  # even user_decision_t means morning decision time
  # odd user_decision_t means evening decision time
  # the way the RL algorithm updates data, the earliest brushing data we get is
  # brushing data associated with the previous days' morning window
  if user_decision_t <= 1:
    return []
  last_available_data_idx = user_decision_t - 2 if (user_decision_t % 2 == 0) else user_decision_t - 3
  return data_df['quality'].values[:last_available_data_idx + 1]

def generate_b_bars(user_df):
  ## init ##
  b_bars = np.zeros(shape=(len(user_df),))

  for dt in range(len(user_df)):
    user_qualities = get_quality_array(user_df, dt)
    b_bar = get_normalize_b_bar(user_qualities)
    b_bars[dt] = b_bar

  return b_bars

# reformatting
ALG_FEATURES_DF = ALG_FEATURES_DF.rename(columns={"timeOfDay": "state_tod"})
ALG_FEATURES_DF["quality"] = ALG_FEATURES_DF['brushingDuration'] - ALG_FEATURES_DF['pressureDuration']
ALG_FEATURES_DF = ALG_FEATURES_DF.drop(columns=['brushingDuration', 'pressureDuration'])

all_b_bars = np.empty(shape=(0,))
for user_id in STUDY_USERS:
  user_df = get_user_df(user_id, ALG_FEATURES_DF)
  b_bars = generate_b_bars(user_df)
  all_b_bars = np.concatenate([all_b_bars, b_bars])

assert len(all_b_bars) == len(ALG_FEATURES_DF)

ALG_FEATURES_DF["state_b_bar"] = all_b_bars
ALG_FEATURES_DF["state_bias"] = np.ones(len(ALG_FEATURES_DF))

ALG_FEATURES_DF

"""## Helpers
---
"""

# grab user specific df
def get_user_df(user_id):
  return ALG_FEATURES_DF[ALG_FEATURES_DF['robas id'] == user_id]

def get_batch_data(user_id):
  user_df = get_user_df(user_id)
  states_df = user_df.filter(regex='state_*')
  rewards = user_df['quality']

  return np.array(states_df), np.array(rewards)

def get_all_users_batch_data(users):
  df = ALG_FEATURES_DF[ALG_FEATURES_DF['robas id'].isin(users)]
  states_df = df.filter(regex='state_*')
  rewards = df['quality']

  return np.array(states_df), np.array(rewards)

# with some regularization
def get_ols_solution(X, Y):
  #   # if matrix is singular then add some noise
  return np.linalg.solve(X.T @ X + 1e-3 * np.diag(np.ones(len(X.T @ X))), X.T @ Y)

"""## Fitting Linear Models
---
ROBAS 3 had no data under action 1 and only had data on 3 features available. Therefore, we fit a linear baseline model with parameters $\theta \in \mathbb{R}^{3}$.
"""

def get_user_lin_model_params(users):
  # fit one theta per user
  thetas = []

  for user_id in users:
    states, rewards = get_batch_data(user_id)
    user_theta = get_ols_solution(states, rewards)
    thetas.append(user_theta)

  return np.array(thetas)

"""## Significance Test For Each Feature
---
"""

def calculate_var_estimator(X, users, dim=D):
  matrix = np.zeros(shape=(dim, dim))
  for user in users:
    states, rewards = get_batch_data(user_id)
    vector = np.array([states[i] * rewards[i] for i in range(len(states))])
    matrix += vector.T @ vector

  return np.linalg.inv(X.T @ X) @ matrix @ np.linalg.inv(X.T @ X)

# This is the cut off value. If the test statistic (calculated above) is greater than the cut off value, then the feature is significiant.
# Cut Off Value = $|inverse CDF (significance / 2, num. of users)|$
def compute_cut_off_value(users):

  return abs(scipy.stats.t.ppf(0.05 / 2, len(users)))

# gets indices of significant features
def conduct_significance_test(users):
  total_X, total_Y = get_all_users_batch_data(users)
  var_matrix = calculate_var_estimator(total_X, users)

  w = np.linalg.inv(total_X.T @ total_X) @ total_X.T @ total_Y

  return [abs(w[i] / var_matrix[i][i]**(0.5)) for i in range(len(w))]

# For features that are significant in the GEE analysis, the prior mean is set to be
# empirical mean of that feature weight across users otherwise
# the prior sd is set to be the empirical standard deviation across user model.
# For features that are not significant, the prior mean is 0 and
# we shrink the empirical standard deviation of that feature by half.
def get_mean_and_sd(users, param_means, param_sds):
  cut_off = compute_cut_off_value(users)
  significance_vals = conduct_significance_test(users)
  significance_idxs = np.array(significance_vals >= cut_off)

  return param_means * significance_idxs, (param_sds / 2) * (1 + significance_idxs)

compute_cut_off_value(STUDY_USERS)

conduct_significance_test(STUDY_USERS)

# means
lin_model_params = get_user_lin_model_params(STUDY_USERS)
param_means = np.mean(lin_model_params, axis=0)
param_sds = np.std(lin_model_params, axis=0)
prior_means, prior_sds = get_mean_and_sd(STUDY_USERS, param_means, param_sds)

param_means, param_sds

prior_means, prior_sds

# std of parameters fitted across users
lin_model_params = get_user_lin_model_params(STUDY_USERS)
print("Prior Means:")
for val in ["{:.3f} \n".format(mean) for mean in prior_means]:
  print(val)
print("Prior Vars:")
for val in ["{:.3f}^2 \n".format(sd) for sd in prior_sds]:
  print(val)

# for features that were not present in the dataset, we set the prior mean to 0,
# and the sd is average sd across all other features.
sd_for_missing_features = np.mean(prior_sds)
print("Prior Var For Missing Features:")
print("{:.3f}^2 \n".format(sd_for_missing_features))

"""## Fitting $\sigma_n^2$
---
1. We fit one linear regression model per user.
2. We then obtain the weights for each fitted model and calculate residuals.
3. $\sigma_n$ is set to the average SD of the residuals.

Closed Form soluation for linear regression:
$w^* = (X^TX)^{-1}X^Ty$
"""

def calculate_noise_var(users):
  # fit one sigma_n per user to find the variance of sigma_n
  sigma_n_squared_s = []

  for user_id in users:
    states, rewards = get_batch_data(user_id)
    user_theta = get_ols_solution(states, rewards)

    user_predicted_Y =  states @ user_theta
    user_residuals = rewards - user_predicted_Y

    sigma_n_squared_s.append(np.var(user_residuals))

  return np.mean(sigma_n_squared_s)

print("Noise variance estimate", calculate_noise_var(STUDY_USERS))
