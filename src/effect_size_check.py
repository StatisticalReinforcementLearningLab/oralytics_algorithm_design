import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

ROBAS_3_STAT_PARAMS_DF = pd.read_csv('sim_env_data/stat_user_models.csv')
ROBAS_3_NON_STAT_PARAMS_DF = pd.read_csv('sim_env_data/non_stat_user_models.csv')
ROBAS_3_DATA = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
ROBAS_3_USERS = ROBAS_3_USERS = np.unique(ROBAS_3_DATA['robas id'])

"""## 1. Generating Data Set
---
"""

BERN_PARAM_TITLES = ['Time.of.Day.Bern', \
                     'Prior.Day.Total.Brush.Time.norm.Bern', \
                     'Day.in.Study.norm.Bern', \
                     'Day.Type.Bern']

Y_PARAM_TITLES = ['Time.of.Day.Y', \
                  'Prior.Day.Total.Brush.Time.norm.Y', \
                  'Day.in.Study.norm.Y', \
                  'Day.Type.Y']

SHRINKAGE_VALUE = 8
# SHRINKAGE_VALUE = 4

# returns dictionary of effect sizes where the key is the user_id
# and the value is a tuple where the tuple[0] is the bernoulli effect size
# and tuple[1] is the effect size on y
# users are grouped by their base model type first when calculating imputed effect sizes
def get_effect_size_on_intercept(parameter_df, bern_param_titles, y_param_titles):
    hurdle_df = parameter_df[parameter_df['Model Type'] == 'sqrt_norm']
    zip_df = parameter_df[parameter_df['Model Type'] == 'zero_infl']
    get_mean_across_features = lambda array: np.mean(np.abs(array), axis=1) / SHRINKAGE_VALUE
    ### HURDLE ###
    hurdle_bern_param_array = np.array([hurdle_df[title] for title in bern_param_titles])
    hurdle_y_param_array = np.array([hurdle_df[title] for title in y_param_titles])
    # effect size bias mean
    hurdle_bern_intercept = 2*np.mean(get_mean_across_features(hurdle_bern_param_array))
    hurdle_y_intercept = 2*np.mean(get_mean_across_features(hurdle_y_param_array))

    ### ZIP ###
    zip_bern_param_array = np.array([zip_df[title] for title in bern_param_titles])
    zip_y_param_array = np.array([zip_df[title] for title in y_param_titles])
    # effect size bias mean
    zip_bern_intercept = 2*np.max(get_mean_across_features(zip_bern_param_array))
    zip_y_intercept = 2*np.max(get_mean_across_features(zip_y_param_array))

    # just the intercept
    user_effect_sizes = {}
    for user in parameter_df['User']:
        if np.array(parameter_df[parameter_df['User'] == user])[0][2] == "sqrt_norm":
            user_effect_sizes[user] = [hurdle_bern_intercept, hurdle_y_intercept]
        else:
            user_effect_sizes[user] = [zip_bern_intercept, zip_y_intercept]

    return user_effect_sizes

STAT_USER_EFFECT_SIZES = get_effect_size_on_intercept(ROBAS_3_STAT_PARAMS_DF, BERN_PARAM_TITLES[:2] + BERN_PARAM_TITLES[3:], \
Y_PARAM_TITLES[:2] + Y_PARAM_TITLES[3:])
NON_STAT_USER_EFFECT_SIZES = get_effect_size_on_intercept(ROBAS_3_NON_STAT_PARAMS_DF, BERN_PARAM_TITLES, Y_PARAM_TITLES)

# generating states
# total brushing quality
robas_3_user_total_brush_quality = (np.array(ROBAS_3_DATA['brushingDuration'])[::2] - np.array(ROBAS_3_DATA['pressureDuration'])[::2])\
 + (np.array(ROBAS_3_DATA['brushingDuration'])[1::2] - np.array(ROBAS_3_DATA['pressureDuration'])[1::2])

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
  # user specific normalization for day in study
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

for user_id in ROBAS_3_USERS:
  user_rewards = get_rewards(user_id)
  users_sessions_stationarity[user_id] = generate_state_spaces_stationarity(user_id, user_rewards)
  users_sessions_non_stationarity[user_id] = generate_state_spaces_non_stationarity(user_id, user_rewards)

def get_params_for_user(user, env_type):
  param_dim = 5 if env_type=='stat' else 6
  param_df = ROBAS_3_STAT_PARAMS_DF if env_type=='stat' else ROBAS_3_NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  model_type = user_row[0][2]
  bern_params = user_row[0][3:3 + param_dim]
  y_params = user_row[0][3 + param_dim: 3 + 2 * param_dim]
  sigma_u = user_row[0][-1]

  # bernouilli parameters, y parameters
  # note: for zip models, sigam_u has a NaN value
  return bern_params, y_params, sigma_u

def construct_model_and_sample(state, action, \
                                          bern_params, \
                                          y_params, \
                                          sigma_u, \
                                          model_type, \
                                          effect_func_bern=lambda state : 0, \
                                          effect_func_y=lambda state : 0):
  bern_linear_comp = state @ bern_params
  if (action == 1):
    # print("EFFECT SIZE B", effect_func_bern(state))
    bern_linear_comp -= effect_func_bern(state)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
      y_mu = state @ y_params
      if (action == 1):
          # print("EFFECT SIZE Y", effect_func_y(state))
          y_mu += effect_func_y(state)
      if model_type == "sqrt_norm":
        # normal transform component
        sample = norm.rvs(loc=y_mu, scale=sigma_u)
        sample = sample**2

        # we round to the nearest integer to produce brushing duration in seconds
        return int(sample)
      else:
        # poisson component
        l = np.exp(y_mu)
        sample = poisson.rvs(l)

        return sample

  else:
    return 0

STAT_USER_REWARD_GENERATING_FUNCTIONS = {}
NON_STAT_USER_REWARD_GENERATING_FUNCTIONS = {}
for user_id in ROBAS_3_USERS:
    stat_user_params = get_params_for_user(user_id, 'stat')
    stat_model_type = np.array(ROBAS_3_STAT_PARAMS_DF[ROBAS_3_STAT_PARAMS_DF['User'] == user_id])[0][2]
    stat_effect_sizes = np.array(STAT_USER_EFFECT_SIZES[user_id])
    non_stat_user_params = get_params_for_user(user_id, 'non_stat')
    non_stat_model_type = np.array(ROBAS_3_NON_STAT_PARAMS_DF[ROBAS_3_NON_STAT_PARAMS_DF['User'] == user_id])[0][2]
    non_stat_effect_sizes = np.array(NON_STAT_USER_EFFECT_SIZES[user_id])
    stat_user_reward_function = lambda state, action: construct_model_and_sample(state, action, \
                                      stat_user_params[0], \
                                      stat_user_params[1], \
                                      stat_user_params[2], \
                                      stat_model_type, \
                                      effect_func_bern=lambda state: stat_effect_sizes[0], \
                                      effect_func_y=lambda state: stat_effect_sizes[1])
    non_stat_user_reward_function = lambda state, action: construct_model_and_sample(state, action, \
                                      non_stat_user_params[0], \
                                      non_stat_user_params[1], \
                                      non_stat_user_params[2], \
                                      non_stat_model_type, \
                                      effect_func_bern=lambda state: non_stat_effect_sizes[0], \
                                      effect_func_y=lambda state: non_stat_effect_sizes[1])
    STAT_USER_REWARD_GENERATING_FUNCTIONS[user_id] = stat_user_reward_function
    NON_STAT_USER_REWARD_GENERATING_FUNCTIONS[user_id] = non_stat_user_reward_function

np.random.seed(1)
# generating rewards
NUM_DECISION_TIMES = 140
STAT_STATES = []
STAT_REWARDS = []
STAT_ACTIONS = []
NON_STAT_STATES = []
NON_STAT_REWARDS = []
NON_STAT_ACTIONS = []

for user_id in ROBAS_3_USERS:
    for idx in np.random.choice(len(users_sessions_stationarity[user_id]), NUM_DECISION_TIMES):
        state = users_sessions_stationarity[user_id][idx]
        for action in range(2):
            STAT_STATES.append(state)
            STAT_ACTIONS.append(action)
            reward = STAT_USER_REWARD_GENERATING_FUNCTIONS[user_id](state, action)
            STAT_REWARDS.append(reward)
    for idx in np.random.choice(len(users_sessions_non_stationarity[user_id]), NUM_DECISION_TIMES):
        state = users_sessions_non_stationarity[user_id][idx]
        for action in range(2):
            NON_STAT_STATES.append(state)
            NON_STAT_ACTIONS.append(action)
            reward = NON_STAT_USER_REWARD_GENERATING_FUNCTIONS[user_id](state, action)
            NON_STAT_REWARDS.append(reward)

"""## 2. Obtaining Standard Effect Size.
---
"""
STAT_PHI = np.hstack([STAT_STATES, np.array(STAT_ACTIONS).reshape(-1, 1)])
STAT_R = np.array(STAT_REWARDS)
NON_STAT_PHI = np.hstack([NON_STAT_STATES, np.array(NON_STAT_ACTIONS).reshape(-1, 1)])
NON_STAT_R = np.array(NON_STAT_REWARDS)

STAT_THETA = np.linalg.inv(STAT_PHI.T @ STAT_PHI) @ STAT_PHI.T @ STAT_R
NON_STAT_THETA = np.linalg.inv(NON_STAT_PHI.T @ NON_STAT_PHI) @ NON_STAT_PHI.T @ NON_STAT_R

print("STAT THETA", STAT_THETA)
print("NON STAT THETA", NON_STAT_THETA)

stat_residuals = STAT_R - (STAT_PHI @ STAT_THETA)
non_stat_residuals = NON_STAT_R - (NON_STAT_PHI @ NON_STAT_THETA)
stat_sigma_res = np.std(stat_residuals)
stat_sigma_rewards = np.std(STAT_R)
non_stat_sigma_res = np.std(non_stat_residuals)
non_stat_sigma_rewards = np.std(NON_STAT_R)

print("STAT STD OF RESIDUALS", stat_sigma_res)
print("NON-STAT STD OF RESIDUALS", non_stat_sigma_res)
print("STAT STD OF REWARDS", stat_sigma_rewards)
print("NON-STAT STD OF REWARDS", non_stat_sigma_rewards)

# standard effect size
stat_standard_eff_res = STAT_THETA[-1] / stat_sigma_res
stat_standard_eff_reward = STAT_THETA[-1] / stat_sigma_rewards
non_stat_standard_eff_res = NON_STAT_THETA[-1] / non_stat_sigma_res
non_stat_standard_eff_reward = NON_STAT_THETA[-1] / non_stat_sigma_rewards

print("STAT STANDARD EFFECT SIZE (RESIDUALS)", stat_standard_eff_res)
print("NON-STAT STANDARD EFFECT SIZE (RESIDUALS)", non_stat_standard_eff_res)
print("STAT STANDARD EFFECT SIZE (REWARDS)", stat_standard_eff_reward)
print("NON-STAT STANDARD EFFECT SIZE (REWARDS)", non_stat_standard_eff_reward)

# number of datapoints
print("NUMBER OF TRAINING DATAPOINTS", len(STAT_PHI))
