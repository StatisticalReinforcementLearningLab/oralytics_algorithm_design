# -*- coding: utf-8 -*-

# pull packages
import pandas as pd
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

PILOT_DATA = pd.read_csv('sim_env_v3_data.csv')
PILOT_USERS = np.unique(PILOT_DATA['user_id'])

def get_user_data(df, user_id):
  return df[df['user_id'] == user_id]

def get_batch_data(df, user_id, env_type='stat'):
  user_df = get_user_data(df, user_id)
  if env_type == 'stat':
    states_df = user_df.filter(regex='state_*').drop(columns=["state_day_in_study"])
  else:
    states_df = user_df.filter(regex='state_*')
  rewards = user_df['quality']
  actions = user_df['action']

  return np.array(states_df), np.array(rewards), np.array(actions)

"""## Model Selection
---
"""

users_stat_states = {}
users_non_stat_states = {}
users_rewards = {}
users_actions = {}
for user_id in PILOT_USERS:
  stat_states, rewards, actions = get_batch_data(PILOT_DATA, user_id, env_type='stat')
  non_stat_states, _, _ = get_batch_data(PILOT_DATA, user_id, env_type='non_stat')
  users_rewards[user_id] = rewards
  users_actions[user_id] = actions
  users_stat_states[user_id] = stat_states
  users_non_stat_states[user_id] = non_stat_states

## Square Root Transform ##
STAT_SQRT_NORM_DF = pd.read_csv("../../sim_env_data/v3_stat_hurdle_model_params.csv")
NON_STAT_SQRT_NORM_DF = pd.read_csv("../../sim_env_data/v3_non_stat_hurdle_model_params.csv")

## Zero Inflated Poisson ##
STAT_ZERO_INFL_POIS_DF = pd.read_csv("../../sim_env_data/v3_stat_zip_model_params.csv")
NON_STAT_ZERO_INFL_POIS_DF = pd.read_csv("../../sim_env_data/v3_non_stat_zip_model_params.csv")

"""### Helpers
---
"""

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_norm_transform_params_for_user(user, df, env_type='stat'):
  param_dim = 6 if env_type == 'stat' else 7
  user_row = np.array(df[df['User'] == user])
  bern_base = user_row[0][2:2 + param_dim]
  bern_adv = user_row[0][2 + param_dim: 2 + 2 * param_dim]
  mean_base = user_row[0][2 + 2 * param_dim:2 + 3 * param_dim]
  mean_adv = user_row[0][2 + 3 * param_dim:2 + 4 * param_dim]
  sigma_u = user_row[0][-1]

  # poisson parameters, bernouilli parameters
  return bern_base, bern_adv, mean_base, mean_adv, sigma_u

def get_zero_infl_params_for_user(user, df, env_type='stat'):
  param_dim = 6 if env_type == 'stat' else 7
  user_row = np.array(df[df['User'] == user])
  bern_base = user_row[0][2:2 + param_dim]
  bern_adv = user_row[0][2 + param_dim: 2 + 2 * param_dim]
  poisson_base = user_row[0][2 + 2 * param_dim:2 + 3 * param_dim]
  poisson_adv = user_row[0][2 + 3 * param_dim:2 + 4 * param_dim]

  # poisson parameters, bernouilli parameters
  return bern_base, bern_adv, poisson_base, poisson_adv

def bern_mean(w_b, delta_b, A, X):
  bern_term = X @ w_b + A * (X @ delta_b)
  mean = 1 - sigmoid(bern_term)

  return mean

def poisson_mean(w_p, delta_p, A, X):
  poisson_term = X @ w_p + A * (X @ delta_p)
  lam = np.exp(poisson_term)

  return lam

def sqrt_norm_mean(w_mu, delta_mu, sigma_u, A, X):
  norm_mu = X @ w_mu + A * (X @ delta_mu)
  mean = sigma_u**2 + norm_mu**2

  return mean

def compute_rmse_sqrt_norm(Xs, As, Ys, w_b, delta_b, w_mu, delta_mu, sigma_u):
  mean_func = lambda x, a: bern_mean(w_b, delta_b, a, x) * sqrt_norm_mean(w_mu, delta_mu, sigma_u, a, x)
  result = np.array([(Ys[i] - mean_func(Xs[i], As[i]))**2 for i in range(len(Xs))])

  return np.sqrt(np.mean(result))

def compute_rmse_zero_infl(Xs, As, Ys, w_b, delta_b, w_p, delta_p):
  mean_func = lambda x, a: bern_mean(w_b, delta_b, a, x) * poisson_mean(w_p, delta_p, a, x)
  result = np.array([(Ys[i] - mean_func(Xs[i], As[i]))**2 for i in range(len(Xs))])
  return np.sqrt(np.mean(result))

"""### Evaluation
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
      As = users_actions[user]
      Ys = users_rewards[user]
      ### HURDLE MODELS ###
      hurdle_w_b, hurdle_delta_b, hurdle_w_mu, hurdle_delta_mu, hurdle_sigma_u = get_norm_transform_params_for_user(user, sqrt_norm_df, env_type)
      user_rmses[0] = compute_rmse_sqrt_norm(Xs, As, Ys, hurdle_w_b, hurdle_delta_b, hurdle_w_mu, hurdle_delta_mu, hurdle_sigma_u)
      ## 0-INFLATED MODELS ##
      zero_infl_w_b, zero_infl_delta_b, zero_infl_w_p, zero_infl_delta_p = get_zero_infl_params_for_user(user, zero_infl_pois_df, env_type)
      user_rmses[1] = compute_rmse_zero_infl(Xs, As, Ys, zero_infl_w_b, zero_infl_delta_b, zero_infl_w_p, zero_infl_delta_p)
      print(MODEL_CLASS[np.argmin(user_rmses)])
      print("RMSES: ", user_rmses)

      USER_MODELS[user] = MODEL_CLASS[np.argmin(user_rmses)]
      USER_RMSES[user] = user_rmses

  return USER_MODELS, USER_RMSES

STAT_USER_MODELS, STAT_USER_RMSES = choose_best_model(users_stat_states, users_rewards, \
                                     STAT_SQRT_NORM_DF, STAT_ZERO_INFL_POIS_DF)

NON_STAT_USER_MODELS, NON_STAT_USER_RMSES = choose_best_model(users_non_stat_states, users_rewards, \
                                     NON_STAT_SQRT_NORM_DF, NON_STAT_ZERO_INFL_POIS_DF, env_type='non_stat')

print("For Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in STAT_USER_MODELS.values() if x == zero_infl]))

print("For Non-Stationary Environment: ")
print("Num. Sqrt Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == sqrt_norm]))
print("Num. 0-Inflated Poisson Model: ", len([x for x in NON_STAT_USER_MODELS.values() if x == zero_infl]))
