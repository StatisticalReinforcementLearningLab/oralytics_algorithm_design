# -*- coding: utf-8 -*-

# pull packages
import pandas as pd
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import norm
from scipy.stats import lognorm
from scipy.stats import poisson
from scipy.stats import kurtosis
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# import theano.tensor as tt
# import arviz as az
# as of Aug. 2022, pymc v4 is out:
# https://discourse.pymc.io/t/google-colab-update-pymc3-is-being-replaced-by-pymc-v4-faq/10235
import pymc as pm
from pymc.model import Model

import pickle

PILOT_DATA = pd.read_csv('https://raw.githubusercontent.com/StatisticalReinforcementLearningLab/oralytics_pilot_data/main/pilot_data_with_feature_engineering.csv')
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

get_user_data(PILOT_DATA, PILOT_USERS[0])

"""## Fitting Models
---

### Helpers
---
"""

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def build_bern_model(X, A, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_b = pm.MvNormal('w_b', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    delta_b = pm.MvNormal('delta_b', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    base_term = X @ w_b
    adv_term = A * (X @ delta_b)
    bern_term = base_term + adv_term
    # Y:= vector of 0s and 1s
    R = pm.Bernoulli('R', p=1 - sigmoid(bern_term), observed=Y)

  return model

def build_sqrt_norm_model(X, A, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_mu = pm.MvNormal('w_mu', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    delta_mu = pm.MvNormal('delta_mu', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    sigma_u = pm.Normal('sigma_u', mu=0, sigma=1)
    mean_term = X @ w_mu + A * (X @ delta_mu)
    # we square the link function as a reparameterization
    R = pm.Normal('R', mu=mean_term, sigma=sigma_u, observed=Y**0.5)

  return model

def build_0_inflated_poisson_model(X, A, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_b = pm.MvNormal('w_b', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    delta_b = pm.MvNormal('delta_b', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    w_p = pm.MvNormal('w_p', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    delta_p = pm.MvNormal('delta_p', mu=np.zeros(d,), cov=np.eye(d), shape=(d,))
    bern_term = X @ w_b + A * (X @ delta_b)
    poisson_term = X @ w_p + A * (X @ delta_p)
    R = pm.ZeroInflatedPoisson("likelihood", psi=1 - sigmoid(bern_term), mu=np.exp(poisson_term), observed=Y)

  return model

def turn_to_bern_labels(Y):
  return np.array([1 if val > 0 else 0 for val in Y])

def turn_to_norm_state_and_labels(X, A, Y):
  idxs = np.where(Y != 0)
  return X[idxs], A[idxs], Y[idxs]

from numpy.core.multiarray import concatenate

def run_bern_map_for_users(users_sessions, users_actions, users_rewards, d, num_restarts):
  model_params = {}
  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_actions = users_actions[user_id]
    user_rewards = users_rewards[user_id]
    user_bern_rewards = turn_to_bern_labels(user_rewards)
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 2 * d))
    for seed in range(num_restarts):
      model = build_bern_model(user_states, user_actions, user_bern_rewards)
      np.random.seed(seed + 100)
      init_params = {'w_b': np.random.randn(d), 'delta_b': np.random.randn(d)}
      # init_params = {'w_b': np.ones(d) * seed}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_b = map_estimate['w_b']
      delta_b = map_estimate['delta_b']
      logp_vals[seed] = model.compile_logp()(map_estimate)
      param_vals[seed] = np.concatenate((w_b, delta_b), axis=0)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

def run_hurdle_map_for_users(users_sessions, users_actions, users_rewards, d, num_restarts):
  model_params = {}
  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_actions = users_actions[user_id]
    user_rewards = users_rewards[user_id]
    transform_norm_states, transform_norm_actions, transform_norm_rewards = turn_to_norm_state_and_labels(user_states, user_actions, user_rewards)
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 2 * d + 1))
    for seed in range(num_restarts):
      # sqrt transform
      model = build_sqrt_norm_model(transform_norm_states, transform_norm_actions, transform_norm_rewards)
      np.random.seed(seed)
      init_params = {'w_mu': np.random.randn(d), 'sigma_u': abs(np.random.randn()), 'delta_mu': np.random.randn(d)}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_mu = map_estimate['w_mu']
      sigma_u = map_estimate['sigma_u']
      delta_mu = map_estimate['delta_mu']
      logp_vals[seed] = model.compile_logp()(map_estimate)
      param_vals[seed] = np.concatenate((w_mu, delta_mu, np.array([sigma_u])), axis=0)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

def run_zero_infl_map_for_users(users_sessions, users_actions, users_rewards, d, num_restarts):
  model_params = {}

  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_actions = users_actions[user_id]
    user_rewards = users_rewards[user_id]
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 4 * d))
    for seed in range(num_restarts):
      model = build_0_inflated_poisson_model(user_states, user_actions, user_rewards)
      np.random.seed(seed)
      init_params = {'w_b': np.random.randn(d), 'delta_b': np.random.randn(d), 'w_p':  np.random.randn(d), 'delta_p': np.random.randn(d)}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_b = map_estimate['w_b']
      delta_b = map_estimate['delta_b']
      w_p = map_estimate['w_p']
      delta_p = map_estimate['delta_p']
      logp_vals[seed] = model.compile_logp()(map_estimate)
      param_vals[seed] = np.concatenate((w_b, delta_b, w_p, delta_p), axis=None)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

"""### Execution
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

# stationary model, bernoulli params
stat_bern_model_params = run_bern_map_for_users(users_stat_states, users_actions, users_rewards, d=6, num_restarts=5)
stat_bern_model_params

# stationary model, hurdle params
stat_hurdle_params = run_hurdle_map_for_users(users_stat_states, users_actions, users_rewards, d=6, num_restarts=5)
stat_hurdle_params

# stationary model, zip params
stat_zip_model_params = run_zero_infl_map_for_users(users_stat_states, users_actions, users_rewards, d=6, num_restarts=5)
stat_zip_model_params

## non-stationary models ##
# bernoulli params
non_stat_bern_model_params = run_bern_map_for_users(users_non_stat_states, users_actions, users_rewards, d=7, num_restarts=5)
non_stat_bern_model_params

# hurdle params
non_stat_hurdle_params = run_hurdle_map_for_users(users_non_stat_states, users_actions, users_rewards, d=7, num_restarts=5)
non_stat_hurdle_params

# zip params
non_stat_zip_model_params = run_zero_infl_map_for_users(users_non_stat_states, users_actions, users_rewards, d=7, num_restarts=5)
non_stat_zip_model_params

"""## Saving Parameter Values
---
"""

# model_columns must contain "User" as the indexing key
def create_hurdle_df_from_params(model_columns, bern_model_params, normal_transform_params):
  df = pd.DataFrame(columns = model_columns)
  for user in bern_model_params.keys():
    bern_vals = bern_model_params[user]
    norm_transform_vals = normal_transform_params[user]
    values = np.concatenate((bern_vals, norm_transform_vals), axis=0)
    new_row = {}
    new_row['User'] = user
    for i in range(1, len(model_columns)):
      new_row[model_columns[i]] = values[i - 1]
    df = df.append(new_row, ignore_index=True)

  return df

def create_zip_df_from_params(model_columns, zip_model_params):
  df = pd.DataFrame(columns = model_columns)
  for user in zip_model_params.keys():
    values = zip_model_params[user]
    new_row = {}
    new_row['User'] = user
    for i in range(1, len(model_columns)):
      new_row[model_columns[i]] = values[i - 1]
    df = df.append(new_row, ignore_index=True)

  return df

stat_hurdle_model_columns = ['User', 'state_tod.Base.Bern', 'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern', 'state_app_engage.Base.Bern', 'state_day_type.Base.Bern', 'state_bias.Base.Bern', \
                                       'state_tod.Adv.Bern', 'state_b_bar.norm.Adv.Bern', 'state_a_bar.norm.Adv.Bern', 'state_app_engage.Adv.Bern', 'state_day_type.Adv.Bern', 'state_bias.Adv.Bern', \
                                       'state_tod.Base.Mu', 'state_b_bar.norm.Base.Mu', 'state_a_bar.norm.Base.Mu', 'state_app_engage.Base.Mu', 'state_day_type.Base.Mu', 'state_bias.Base.Mu', \
                                       'state_tod.Adv.Mu', 'state_b_bar.norm.Adv.Mu', 'state_a_bar.norm.Adv.Mu', 'state_app_engage.Adv.Mu', 'state_day_type.Adv.Mu', 'state_bias.Adv.Mu', \
                                      'Sigma_u']

non_stat_hurdle_model_columns = ['User', 'state_tod.Base.Bern', 'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern', 'state_app_engage.Base.Bern', 'state_day_type.Base.Bern', 'state_bias.Base.Bern', 'state_day_in_study.Base.Bern', \
                                       'state_tod.Adv.Bern', 'state_b_bar.norm.Adv.Bern', 'state_a_bar.norm.Adv.Bern', 'state_app_engage.Adv.Bern', 'state_day_type.Adv.Bern', 'state_bias.Adv.Bern', 'state_day_in_study.Adv.Bern', \
                                       'state_tod.Base.Mu', 'state_b_bar.norm.Base.Mu', 'state_a_bar.norm.Base.Mu', 'state_app_engage.Base.Mu', 'state_day_type.Base.Mu', 'state_bias.Base.Mu', 'state_day_in_study.Base.Mu', \
                                       'state_tod.Adv.Mu', 'state_b_bar.norm.Adv.Mu', 'state_a_bar.norm.Adv.Mu', 'state_app_engage.Adv.Mu', 'state_day_type.Adv.Mu', 'state_bias.Adv.Mu', 'state_day_in_study.Adv.Mu', \
                                      'Sigma_u']

stat_zip_model_columns = ['User', 'state_tod.Base.Bern', 'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern', 'state_app_engage.Base.Bern', 'state_day_type.Base.Bern', 'state_bias.Base.Bern', \
                                       'state_tod.Adv.Bern', 'state_b_bar.norm.Adv.Bern', 'state_a_bar.norm.Adv.Bern', 'state_app_engage.Adv.Bern', 'state_day_type.Adv.Bern', 'state_bias.Adv.Bern', \
                                       'state_tod.Base.Poisson', 'state_b_bar.norm.Base.Poisson', 'state_a_bar.norm.Base.Poisson', 'state_app_engage.Base.Poisson', 'state_day_type.Base.Poisson', 'state_bias.Base.Poisson', \
                                       'state_tod.Adv.Poisson', 'state_b_bar.norm.Adv.Poisson', 'state_a_bar.norm.Adv.Poisson', 'state_app_engage.Adv.Poisson', 'state_day_type.Adv.Poisson', 'state_bias.Adv.Poisson']

non_stat_zip_model_columns = ['User', 'state_tod.Base.Bern', 'state_b_bar.norm.Base.Bern', 'state_a_bar.norm.Base.Bern', 'state_app_engage.Base.Bern', 'state_day_type.Base.Bern', 'state_bias.Base.Bern', 'state_day_in_study.Base.Bern', \
                                       'state_tod.Adv.Bern', 'state_b_bar.norm.Adv.Bern', 'state_a_bar.norm.Adv.Bern', 'state_app_engage.Adv.Bern', 'state_day_type.Adv.Bern', 'state_bias.Adv.Bern', 'state_day_in_study.Adv.Bern', \
                                       'state_tod.Base.Poisson', 'state_b_bar.norm.Base.Poisson', 'state_a_bar.norm.Base.Poisson', 'state_app_engage.Base.Poisson', 'state_day_type.Base.Poisson', 'state_bias.Base.Poisson', 'state_day_in_study.Base.Poisson', \
                                       'state_tod.Adv.Poisson', 'state_b_bar.norm.Adv.Poisson', 'state_a_bar.norm.Adv.Poisson', 'state_app_engage.Adv.Poisson', 'state_day_type.Adv.Poisson', 'state_bias.Adv.Poisson', 'state_day_in_study.Adv.Poisson']

stat_hurdle_df = create_hurdle_df_from_params(stat_hurdle_model_columns, stat_bern_model_params, stat_hurdle_params)

stat_zip_df = create_zip_df_from_params(stat_zip_model_columns, stat_zip_model_params)

stat_hurdle_df.to_csv('../../sim_env_data/v3_stat_hurdle_model_params.csv')
stat_zip_df.to_csv('../../sim_env_data/v3_stat_zip_model_params.csv')

non_stat_hurdle_df = create_hurdle_df_from_params(non_stat_hurdle_model_columns, non_stat_bern_model_params, non_stat_hurdle_params)

non_stat_zip_df = create_zip_df_from_params(non_stat_zip_model_columns, non_stat_zip_model_params)

non_stat_hurdle_df.to_csv('../../sim_env_data/v3_non_stat_hurdle_model_params.csv')
non_stat_zip_df.to_csv('../../sim_env_data/v3_non_stat_zip_model_params.csv')
