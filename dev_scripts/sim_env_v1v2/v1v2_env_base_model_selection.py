# -*- coding: utf-8 -*-
"""
Selecting one model out of the three base model classes based on RMSE
"""

# pull packages
import numpy as np
import pandas as pd


from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""## Creating the State Space and Reward From ROBAS 3 Data
---
"""

# get all robas 3 users
ROBAS_3_DATA = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
ROBAS_3_USERS = ROBAS_3_USERS = np.unique(ROBAS_3_DATA['robas id'])

# total brushing quality
robas_3_user_total_brush_quality = (np.array(ROBAS_3_DATA['brushingDuration'])[::2] - np.array(ROBAS_3_DATA['pressureDuration'])[::2])\
 + (np.array(ROBAS_3_DATA['brushingDuration'])[1::2] - np.array(ROBAS_3_DATA['pressureDuration'])[1::2])

print("Empirical Mean: ", np.mean(robas_3_user_total_brush_quality))
print("Empirical Std: ", np.std(robas_3_user_total_brush_quality))

# Z-score normalization
def normalize_total_brush_quality(quality):
  return (quality - np.mean(robas_3_user_total_brush_quality)) / np.std(robas_3_user_total_brush_quality)

# for the expected 70 day study
def normalize_day_in_study(day):
  return (day - 35.5) / 34.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_rewards(user_id):
  return np.array(ROBAS_3_DATA[ROBAS_3_DATA['robas id'] == user_id]['brushingDuration'] - \
                  ROBAS_3_DATA[ROBAS_3_DATA['robas id'] == user_id]['pressureDuration'])[:140]

# only get the first 70 days when fitting the environment model
def get_user_df(user_id):
  return ROBAS_3_DATA[ROBAS_3_DATA['robas id'] == user_id][:140]

# generating stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Prop. Non-Zero Brushing In Past 7 Days
# 3 - Weekday vs. Weekend
# 4 - Bias

# 0 - Unnamed: 0	1 - ROBAS ID	2- Day in Study	3 - Time of Day	4 - Brushing duration	5 - Pressure duration	6 - Day Type	7 - Proportion Brushed In Past 7 Days
def generate_state_spaces_stationarity(user_id, rewards):
  ## init ##
  D = 5
  user_df = get_user_df(user_id)
  states = np.zeros(shape=(len(user_df), D))
  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[3]
    # prior day brush time at same time of day
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_quality(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_quality(rewards[i - 2] + rewards[i - 3])
    # prop. brushed in past 7 days
    if i > 13:
      states[i][2] = df_array[7]
    # weekday or weekend term
    states[i][3] = df_array[6]
    # bias term
    states[i][4] = 1

  return states


# generating non-stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Day In Study
# 3 - Prop. Non-Zero Brushing In Past 7 Days
# 4 - Weekday vs. Weekend
# 5 - Bias

# 0 - Unnamed: 0	1 - ROBAS ID	2- Day in Study	3 - Time of Day	4 - Brushing duration	5 - Pressure duration	6 - Day Type	7 - Proportion Brushed In Past 7 Days
def generate_state_spaces_non_stationarity(user_id, rewards):
  ## init ##
  D = 6
  user_df = get_user_df(user_id)
  states = np.zeros(shape=(len(user_df), D))
  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[3]
    # prior day brush time at same time of day
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_quality(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_quality(rewards[i - 2] + rewards[i - 3])
    # day in study
    states[i][2] = normalize_day_in_study(df_array[2])
    # prop. brushed in past 7 days
    if i > 13:
      states[i][3] = df_array[7]
    # weekday or weekend term
    states[i][4] = df_array[6]
    # bias term
    states[i][5] = 1

  return states

# generate N=32 users
NUM_USERS = len(ROBAS_3_USERS)

# dictionary where key is user id and values are lists of sessions of trial
users_sessions_stationarity = {}
users_sessions_non_stationarity = {}
users_rewards = {}
for user_id in ROBAS_3_USERS:
  user_rewards = get_rewards(user_id)
  users_rewards[user_id] = user_rewards
  users_sessions_stationarity[user_id] = generate_state_spaces_stationarity(user_id, user_rewards)
  users_sessions_non_stationarity[user_id] = generate_state_spaces_non_stationarity(user_id, user_rewards)

"""## Model Selection
---
"""

## Square Root Transform ##
STAT_SQRT_NORM_DF = pd.read_csv("../../sim_env_data/stat_sqrt_norm_hurdle_model_params.csv")
NON_STAT_SQRT_NORM_DF = pd.read_csv("../../sim_env_data/non_stat_sqrt_norm_hurdle_model_params.csv")

## Zero Inflated Poisson ##
STAT_ZERO_INFL_POIS_DF = pd.read_csv("../../sim_env_data/stat_zip_model_params.csv")
NON_STAT_ZERO_INFL_POIS_DF = pd.read_csv("../../sim_env_data/non_stat_zip_model_params.csv")

def get_norm_transform_params_for_user(user, df, env_type='stat'):
  param_dim = 5 if env_type == 'stat' else 6
  user_row = np.array(df[df['User'] == user])
  bern_params = user_row[0][2:2 + param_dim]
  normal_mean_params = user_row[0][2 + param_dim:2 + 2 * param_dim]
  sigma_u = user_row[0][-1]

  # poisson parameters, bernouilli parameters
  return bern_params, normal_mean_params, sigma_u

def get_zero_infl_params_for_user(user, df, env_type='stat'):
  param_dim = 5 if env_type == 'stat' else 6
  user_row = np.array(df[df['User'] == user])
  bern_params = user_row[0][2:2 + param_dim]
  poisson_params = user_row[0][2 + param_dim:]

  # poisson parameters, bernouilli parameters
  return bern_params, poisson_params

def bern_mean(w_b, X):
  q = sigmoid(X @ w_b)
  mean = 1 - q

  return mean

def poisson_mean(w_p, X):
  lam = np.exp(X @ w_p)

  return lam

def sqrt_norm_mean(w_mu, sigma_u, X):
  norm_mu = X @ w_mu
  mean = sigma_u**2 + norm_mu**2

  return mean

def compute_rmse_sqrt_norm(Xs, Ys, w_b, w_mu, sigma_u):
  mean_func = lambda x: bern_mean(w_b, x) * sqrt_norm_mean(w_mu, sigma_u, x)
  result = np.array([(Ys - mean_func(X))**2 for X in Xs])

  return np.sqrt(np.mean(result))

def compute_rmse_zero_infl(Xs, Ys, w_b, w_p):
  mean_func = lambda x: bern_mean(w_b, x) * poisson_mean(w_p, x)
  result = np.array([(Ys - mean_func(X))**2 for X in Xs])
  return np.sqrt(np.mean(result))

"""### Evaluations
---
"""

# sqrt, zip
sqrt_norm = 'sqrt_norm'
zero_infl = 'zero_infl'

MODEL_CLASS = {0: sqrt_norm, 1: zero_infl}

def choose_best_model(users_dict, users_rewards, sqrt_norm_df, zero_infl_pois_df, env_type='stat'):
  USER_MODELS = {}
  USER_RMSES = {}

  for i, user in enumerate(users_dict.keys()):
      print("FOR USER: ", user)
      user_rmses = np.zeros(2)
      Xs = users_dict[user]
      Ys = users_rewards[user]
      ### HURDLE MODELS ###
      hurdle_w_b = get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[0]
      ## SQRT TRANSFORM ##
      sqrt_w_mu = get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[1]
      sqrt_sigma_u = abs(get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)[2])
      user_rmses[0] = compute_rmse_sqrt_norm(Xs, Ys, hurdle_w_b, sqrt_w_mu, sqrt_sigma_u)
      ## 0-INFLATED MODELS ##
      zero_infl_w_b = get_zero_infl_params_for_user(user, zero_infl_pois_df, env_type)[0]
      w_p = get_zero_infl_params_for_user(user, zero_infl_pois_df, env_type)[1]
      user_rmses[1] = compute_rmse_zero_infl(Xs, Ys, zero_infl_w_b, w_p)
      print(MODEL_CLASS[np.argmin(user_rmses)])
      print("RMSES: ", user_rmses)

      USER_MODELS[user] = MODEL_CLASS[np.argmin(user_rmses)]
      USER_RMSES[user] = user_rmses

  return USER_MODELS, USER_RMSES

STAT_USER_MODELS, STAT_USER_RMSES = choose_best_model(users_sessions_stationarity, users_rewards, \
                                     STAT_SQRT_NORM_DF, STAT_ZERO_INFL_POIS_DF)

NON_STAT_USER_MODELS, NON_STAT_USER_RMSES = choose_best_model(users_sessions_non_stationarity, users_rewards, \
                                     NON_STAT_SQRT_NORM_DF, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')

print("For Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in STAT_USER_MODELS.values() if x == zero_infl]))

print("For Non-Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == zero_infl]))

"""### Saving Parameter Values Into CSV File
---
"""

def create_df_from_params(user_best_model_type, model_columns, hurdle_params_df, zip_model_params_df, env_type):
  df = pd.DataFrame(columns = model_columns)
  for user in user_best_model_type.keys():
    model_type = user_best_model_type[user]
    values = get_norm_transform_params_for_user(user, hurdle_params_df, env_type) \
    if model_type == 'sqrt_norm' else get_zero_infl_params_for_user(user, zip_model_params_df, env_type)
    loop_stop = len(model_columns) if model_type == 'sqrt_norm' else len(model_columns) - 1
    values = np.hstack(values)
    new_row = {}
    new_row['User'] = user
    new_row['Model Type'] = model_type
    for i in range(2, loop_stop):
      new_row[model_columns[i]] = values[i - 2]
    df = df.append(new_row, ignore_index=True)

  return df

stat_env_model_columns = ['User', 'Model Type', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Y', 'Prior.Day.Total.Brush.Time.norm.Y', 'Proportion.Brushed.In.Past.7.Days.Y', 'Day.Type.Y', 'Intercept.Y', \
                        'Sigma_u']

non_stat_env_model_columns = ['User', 'Model Type', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Day.in.Study.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Y', 'Prior.Day.Total.Brush.Time.norm.Y', 'Day.in.Study.norm.Y', 'Proportion.Brushed.In.Past.7.Days.Y', 'Day.Type.Y', 'Intercept.Y',
                        'Sigma_u']

# pickling dicts
stat_env_models_df = create_df_from_params(STAT_USER_MODELS, stat_env_model_columns, STAT_SQRT_NORM_DF, STAT_ZERO_INFL_POIS_DF, 'stat')
non_stat_env_models_df = create_df_from_params(NON_STAT_USER_MODELS, non_stat_env_model_columns, NON_STAT_SQRT_NORM_DF, NON_STAT_ZERO_INFL_POIS_DF, 'non_stat')

stat_env_models_df.to_csv("../../sim_env_data/stat_user_models.csv")
non_stat_env_models_df.to_csv("../../sim_env_data/non_stat_user_models.csv")
