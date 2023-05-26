# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

import read_write_info
import simulation_environment

"""## BASE ENVIRONMENT COMPONENT
---
"""

STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/stat_user_models.csv')
NON_STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/non_stat_user_models.csv')
DATA_DF = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
SIM_ENV_USERS = np.array(STAT_PARAMS_DF['User'])
# value used by run_experiments to draw with replacement
NUM_USER_MODELS = len(SIM_ENV_USERS)

### MODEL TYPE ###
def get_user_model_type(user_id, env_type):
    if env_type == "STAT":
        return np.array(STAT_PARAMS_DF[STAT_PARAMS_DF['User'] == user_id])[0][2]
    else:
        return np.array(NON_STAT_PARAMS_DF[NON_STAT_PARAMS_DF['User'] == user_id])[0][2]

### STATE SPACE ###
def get_user_df(user_id):
  return DATA_DF[DATA_DF['robas id'] == user_id]

# Stationary State Space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Prop. Non-Zero Brushing In Past 7 Days
# 3 - Weekday vs. Weekend
# 4 - Bias
def generate_state_spaces_stat(user_df, num_days):
  ## init ##
  D = 5
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # bias term
    states[i][4] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['dayType']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,3] = 1

  return states

# Non-stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Day In Study
# 3 - Prop. Non-Zero Brushing In Past 7 Days
# 4 - Weekday vs. Weekend
# 5 - Bias
def generate_state_spaces_non_stat(user_df, num_days):
  ## init ##
  D = 6
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # day in study
    states[i][2] = simulation_environment.normalize_day_in_study(i // 2 + 1)
    # bias term
    states[i][5] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['dayType']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,4] = 1

  return states

### ENVIRONMENT AND ALGORITHM STATE SPACE FUNCTIONS ###
def get_previous_day_total_brush_quality(Qs, time_of_day, j):
    if j > 1:
        if time_of_day == 0:
            return Qs[j - 1] + Qs[j - 2]
        else:
            return Qs[j - 2] + Qs[j - 3]
    # first day there is no brushing
    else:
        return 0

def process_env_state(session, j, Qs, env_type='STAT'):
    env_state = session.copy()
    # session type - either 0 or 1
    session_type = int(env_state[0])
    # update previous day total brush time
    previous_day_total_rewards = get_previous_day_total_brush_quality(Qs, session_type, j)
    env_state[1] = simulation_environment.normalize_total_brush_quality(previous_day_total_rewards)
    # proportion of past success brushing
    prior_idx = 2 if env_type == 'STAT' else 3
    if (j >= 14):
      env_state[prior_idx] = np.mean([Qs[-14:] > 0.0])

    return env_state

"""### Generate States
---
"""

NUM_DAYS = 70
# dictionary where key is index and value is user_id
USER_INDICES = {}

# dictionary where key is user id and values are lists of sessions of trial
USERS_SESSIONS_STAT = {}
USERS_SESSIONS_NON_STAT = {}
for i, user_id in enumerate(SIM_ENV_USERS):
  USER_INDICES[i] = user_id
  user_df = get_user_df(user_id)
  USERS_SESSIONS_STAT[user_id] = generate_state_spaces_stat(user_df, NUM_DAYS)
  USERS_SESSIONS_NON_STAT[user_id] = generate_state_spaces_non_stat(user_df, NUM_DAYS)

def get_user_sessions(user_id, env_type):
    return USERS_SESSIONS_STAT[user_id] if env_type == 'STAT' else USERS_SESSIONS_NON_STAT[user_id]

def get_params_for_user(user, env_type):
  param_dim = 5 if env_type == 'STAT' else 6
  param_df = STAT_PARAMS_DF if env_type=='STAT' else NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  model_type = user_row[0][2]
  bern_params = user_row[0][3:3 + param_dim]
  y_params = user_row[0][3 + param_dim: 3 + 2 * param_dim]
  sigma_u = user_row[0][-1]

  # bernouilli parameters, y parameters
  # note: for zip models, sigam_u has a NaN value
  return bern_params, y_params, sigma_u

"""## TREATMENT EFFECT COMPONENT
---
Effect size are imputed using the each environment's (stationary/non-stationary) fitted parameters
"""

with open(read_write_info.READ_PATH_PREFIX + 'sim_env_data/smaller_stat_user_effect_sizes.p', 'rb') as f:
    SMALLER_STAT_USER_EFFECT_SIZES = pickle.load(f)
with open(read_write_info.READ_PATH_PREFIX + 'sim_env_data/smaller_non_stat_user_effect_sizes.p', 'rb') as f:
    SMALLER_NON_STAT_USER_EFFECT_SIZES = pickle.load(f)
with open(read_write_info.READ_PATH_PREFIX + 'sim_env_data/less_small_stat_user_effect_sizes.p', 'rb') as f:
    LESS_SMALL_STAT_USER_EFFECT_SIZES = pickle.load(f)
with open(read_write_info.READ_PATH_PREFIX + 'sim_env_data/less_small_non_stat_user_effect_sizes.p', 'rb') as f:
    LESS_SMALL_NON_STAT_USER_EFFECT_SIZES = pickle.load(f)

def choose_effect_sizes(user_id, effect_size_scale, env_type):
    if env_type == 'STAT':
        if effect_size_scale == 'smaller':
            return np.array(SMALLER_STAT_USER_EFFECT_SIZES[user_id])
        elif effect_size_scale == 'small':
            return np.array(LESS_SMALL_STAT_USER_EFFECT_SIZES[user_id])
    elif env_type == 'NON_STAT':
        if effect_size_scale == 'smaller':
            return np.array(SMALLER_NON_STAT_USER_EFFECT_SIZES[user_id])
        elif effect_size_scale == 'small':
            return np.array(LESS_SMALL_NON_STAT_USER_EFFECT_SIZES[user_id])

"""
Features interacting with effect sizes:
Note: stationary environments do not have "Day in Study"
---
* Time of Day
* Prior Day Total Brushing Quality
* Day in Study
* Weekend vs. Weekday
* Bias
"""

## USER-SPECIFIC EFFECT SIZES ##
# Context-Aware with all features same as baseline features excpet for Prop. Non-Zero Brushing In Past 7 Days
# which is of index 2 for stat models and of index 3 for non stat models
stat_user_spec_effect_func_bern = lambda state, effect_sizes: -1.0*max(effect_sizes @ np.delete(state, 2), 0)
stat_user_spec_effect_func_y = lambda state, effect_sizes: max(effect_sizes @ np.delete(state, 2), 0)
non_stat_user_spec_effect_func_bern = lambda state, effect_sizes: -1.0*max(effect_sizes @ np.delete(state, 3), 0)
non_stat_user_spec_effect_func_y = lambda state, effect_sizes: max(effect_sizes @ np.delete(state, 3), 0)

def get_user_effect_funcs(env_type):
    if env_type == "STAT":
        return stat_user_spec_effect_func_bern, stat_user_spec_effect_func_y
    else:
        return non_stat_user_spec_effect_func_bern, non_stat_user_spec_effect_func_y

"""## Creating Simulation Environment Objects
---
"""

def create_user_envs(users_list, effect_size_scale, delayed_effect_scale_val, env_type):
    all_user_envs = {}
    for i, user_id in enumerate(users_list):
      model_type = get_user_model_type(user_id, env_type)
      user_sessions = get_user_sessions(user_id, env_type)
      user_effect_sizes = choose_effect_sizes(user_id, effect_size_scale, env_type)
      user_params = get_params_for_user(user_id, env_type)
      user_effect_func_bern, user_effect_func_y = get_user_effect_funcs(env_type)
      new_user = simulation_environment.UserEnvironment(user_id, model_type, user_sessions, user_effect_sizes, \
                delayed_effect_scale_val, user_params, user_effect_func_bern, user_effect_func_y)
      all_user_envs[i] = new_user

    return all_user_envs

class SimulationEnvironmentV1(simulation_environment.SimulationEnvironment):
    def __init__(self, users_list, env_type, effect_size_scale, delayed_effect_scale):
        delayed_effect_scale_val = simulation_environment.get_delayed_effect_scale(delayed_effect_scale)
        user_envs = create_user_envs(users_list, effect_size_scale, delayed_effect_scale_val, env_type)

        super(SimulationEnvironmentV1, self).__init__(users_list, user_envs, env_type)

        self.version = "V1"

    def generate_current_state(self, user_idx, j):
        user_state = self.get_states_for_user(user_idx)[j]
        brushing_qualities = np.array(self.get_env_history(user_idx, "outcomes"))

        return process_env_state(user_state, j, brushing_qualities, self.get_env_type())
