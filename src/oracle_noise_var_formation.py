# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

import pymc3 as pm
from pymc3.model import Model
import theano.tensor as tt
import arviz as az

"""
# Calculating Oracle Noise Variance Using ROBAS 3 Data
---
"""
df = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
alg_features_df = df[['robas id', 'timeOfDay', 'brushingDuration', 'dayType']]
NUM_USERS = len(alg_features_df['robas id'].unique())
# num. of baseline featuers minus A bar (which we don't have data on)
D = 4

# normalized values derived from ROBAS 2 dataset
def normalize_total_brush_time(time):
  return (time - 172) / 118

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

# grab user specific df
def get_user_df(user_id):
  return alg_features_df[alg_features_df['robas id'] == user_id]

def generate_state_spaces_for_single_user(user_id, truncated_rewards):
  ## init ##
  user_df = get_user_df(user_id)
  states = np.zeros(shape=(len(user_df), D))

  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[1]
    # b bar
    if i > 14:
      b_bar = calculate_b_bar(truncated_rewards[i - 14:i])
      states[i][1] = normalize_b_bar(b_bar)
    ### THIS IS THE WAY I HAVE BEEN IMPUTING BUT THIS CAN CHANGE
    elif i > 0 and i < 14:
      pseudo_b_bar = np.mean(truncated_rewards[:i])
      states[i][1] = normalize_b_bar(pseudo_b_bar)
    else:
      states[i][1] = normalize_b_bar(0)
    # weekday or weekend term
    states[i][2] = df_array[3]
    # bias term
    states[i][3] = 1

  return states
  
## making a dictionary of user, trajectory (states, rewards)
# dictionary where key is user id and values are lists of sessions of trial
users_sessions = {}
total_X = np.empty(shape=(1, D))
total_Y = np.empty(shape=1)
for user_id in alg_features_df['robas id'].unique():
  rewards = np.array(alg_features_df.loc[alg_features_df['robas id'] == user_id]['brushingDuration'])
  truncated_rewards = np.array([min(x, 180) for x in rewards])
  states = generate_state_spaces_for_single_user(user_id, truncated_rewards)
  total_X = np.concatenate((total_X, states), axis=0)
  total_Y = np.concatenate((total_Y, rewards), axis=None)
  users_sessions[user_id] = [states, rewards]

total_X = total_X[1:,]
total_Y = total_Y[1:]

"""## Fitting Noise Variance
---
1. We fit one linear regression model per user.
2. We then obtain the weights for each fitted model and calculate residuals.
3. $\sigma_n$ is set to the average SD of the residuals.

Closed Form soluation for linear regression:
$w^* = (X^TX)^{-1}X^Ty$
"""

# fit one sigma_n per user to find the variance of sigma_n
sigma_n_squared_s = []

for user in users_sessions.keys():
  states, rewards = users_sessions[user]
  user_w = np.linalg.inv(states.T @ states) @ states.T @ rewards

  user_predicted_Y =  states @ user_w
  user_residuals = rewards - user_predicted_Y

  sigma_n_squared_s.append(np.var(user_residuals))

print(np.mean(sigma_n_squared_s))

"""## Fitting $\mu_0, \Sigma_0$
---
We fit our prior parameters in accordance to the procedure in [[Liao et. al., 2015]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8439432/#R23) Section 6.3.
1. We use GEE regression anaylsis to fit a model per user to identify significance of each feature.
2. For features that are significant, we set prior mean (mean across users) to be the point estimate from this analysis. Also, prior SD is the empirical SD across participant models (across users) from participant-specific GEE analysis.

  For non-significant features, we set the prior mean to be 0 and the SD is shrunk in half.

$\Sigma_0$ is a diagonal matrix whose diagonal entries are the correspnoding variances for each feature.

Variance estimator for $w^*$:

$(X^\top X)^{-1} (\sum_{i=1}^n \left\{ \sum_{t=1}^T X_{i,t} r_{i,t} \right\}^{\otimes 2} ) (X^\top X)^{-1}$

Note: $(\sum_{i=1}^n \left\{ \sum_{t=1}^T X_{i,t} r_{i,t} \right\}^{\otimes 2} ) \in R^{dxd}$, $\otimes 2$ denotes outer product

### 1. Significance Test For Each Feature
---
"""

def calculate_var_estimator(X):
  matrix = np.zeros(shape=(4, 4))
  for user in users_sessions.keys():
    user_state = users_sessions[user][0]
    user_rewards = users_sessions[user][1]
    vector = np.array([user_state[i] * user_rewards[i] for i in range(len(user_state))])
    matrix += vector.T @ vector

  return np.linalg.inv(X.T @ X) @ matrix @ np.linalg.inv(X.T @ X)

var_matrix = calculate_var_estimator(total_X)

w = np.linalg.inv(total_X.T @ total_X) @ total_X.T @ total_Y

[abs(w[i] / var_matrix[i][i]**(0.5)) for i in range(len(w))]

"""This is the cut off value. If the test statistic (calculated above) is greater than the cut off value, then the feature is significiant.

Cut Off Value = $|inverse CDF (significance / 2, num. of users)|$
"""

abs(scipy.stats.t.ppf(0.05 / 2, NUM_USERS))

"""### Step 2. """

ws = []
for user in users_sessions.keys():
  states, rewards = users_sessions[user]
  user_w = np.linalg.inv(states.T @ states) @ states.T @ rewards
  ws.append(user_w)

# std of ws fitted across users
sds = np.std(ws, axis=0)
### diagonal of Sigma_{\alpha_0} ###
print("diag of Sigma_{alpha_0}: ", [sds[0] / 2, sds[1], sds[2] / 2, sds[3]])

mus = np.mean(ws, axis=0)
### mu_{\alpha_0} ###
print("mu_{alpha}: ",[0, mus[1], 0, mus[3]])

### sigma_{\beta} ###
print("sigma_{beta}: ", np.mean([sds[0] / 2, sds[1], sds[2] / 2, sds[3]]))

"""## Informing Design Of Reward. """
### upper 95th percentile of user brushing informed from ROBAS 2
total_truncated_brush_times = np.array([min(x, 120) for x in alg_features_df['brushingDuration']])
print("Upper 50th Percentile Brushing Duration: ", np.percentile(total_truncated_brush_times, 50))

print("SANITY CHECK!!", np.mean(total_truncated_brush_times))
