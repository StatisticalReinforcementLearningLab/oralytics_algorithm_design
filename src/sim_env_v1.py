# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

import read_write_info
import simulation_environment

"""## BASE ENVIRONMENT COMPONENT
---
"""

ROBAS_3_STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/stat_user_models.csv')
ROBAS_3_NON_STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/non_stat_user_models.csv')
ROBAS_3_DATA_DF = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data_complete.csv")
ROBAS_3_USERS = np.array(ROBAS_3_STAT_PARAMS_DF['User'])
# value used by run_experiments to draw with replacement
NUM_USER_MODELS = len(ROBAS_3_USERS)

### STATE SPACE ###
def get_user_df(user_id):
  return ROBAS_3_DATA_DF[ROBAS_3_DATA_DF['robas id'] == user_id]

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
    previous_day_total_rewards = get_previous_day_total_brush_quality(Qs, session[0], j)
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
for i, user_id in enumerate(ROBAS_3_USERS):
  user_idx = i
  USER_INDICES[user_idx] = user_id
  user_df = get_user_df(user_id)
  USERS_SESSIONS_STAT[user_id] = generate_state_spaces_stat(user_df, NUM_DAYS)
  USERS_SESSIONS_NON_STAT[user_id] = generate_state_spaces_non_stat(user_df, NUM_DAYS)

def get_params_for_user(user, env_type):
  param_dim = 5 if env_type=='STAT' else 6
  param_df = ROBAS_3_STAT_PARAMS_DF if env_type=='STAT' else ROBAS_3_NON_STAT_PARAMS_DF
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


"""## Creating Simulation Environment Objects
---
"""
def choose_effect_sizes(effect_size_scale, env_type):
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

class UserEnvironment():
    def __init__(self, user_id, effect_size_scale, env_type, delayed_effect_scale_val):
        self.user_id = user_id
        self.model_type = np.array(ROBAS_3_STAT_PARAMS_DF[ROBAS_3_STAT_PARAMS_DF['User'] == user_id])[0][2] \
        if env_type == 'STAT' else np.array(ROBAS_3_NON_STAT_PARAMS_DF[ROBAS_3_NON_STAT_PARAMS_DF['User'] == user_id])[0][2]
        # vector: size (T, D) where D = 6 is the dimension of the env. state
        # T is the length of the study
        self.user_states = USERS_SESSIONS_STAT[user_id] if env_type == 'STAT' else USERS_SESSIONS_NON_STAT[user_id]
        # tuple: float values of effect size on bernoulli, poisson components
        self.og_user_effect_sizes = choose_effect_sizes(effect_size_scale, env_type)
        self.user_effect_sizes = np.copy(self.og_user_effect_sizes)
        # float: unresponsive scaling value
        self.delayed_effect_scale_val = delayed_effect_scale_val
        # probability of becoming unresponsive
        ### SETTING TO 1 FOR NOW!
        self.unresponsive_prob = 1.0
        # you can only shrink at most once a week
        self.times_shrunk = 0
        # reward generating function
        self.user_params = get_params_for_user(user_id, env_type)
        self.user_effect_func_bern = stat_user_spec_effect_func_bern if env_type == 'STAT' else non_stat_user_spec_effect_func_bern
        self.user_effect_func_y = stat_user_spec_effect_func_y if env_type == 'STAT' else non_stat_user_spec_effect_func_y
        self.reward_generating_func = lambda state, action: simulation_environment.construct_model_and_sample(user_id, state, action, \
                                          self.user_params[0], \
                                          self.user_params[1], \
                                          self.user_params[2], \
                                          self.model_type, \
                                          effect_func_bern=lambda state: self.user_effect_func_bern(state, self.user_effect_sizes[0]), \
                                          effect_func_y=lambda state: self.user_effect_func_y(state, self.user_effect_sizes[1]))

    def generate_reward(self, state, action):
        return self.reward_generating_func(state, action)

    def update_responsiveness(self, a1_cond, a2_cond, b_cond, j):
        # it's been atleast a week since we last shrunk
        if j % 14 == 0:
            if (b_cond and a1_cond) or a2_cond:
                # draw
                is_unresponsive = bernoulli.rvs(self.unresponsive_prob)
                if is_unresponsive:
                    self.user_effect_sizes = self.user_effect_sizes * self.delayed_effect_scale_val
                    self.times_shrunk += 1

            elif self.times_shrunk > 0:
                if self.delayed_effect_scale_val == 0:
                    self.user_effect_sizes[0] = self.og_user_effect_sizes[0]
                    self.user_effect_sizes[1] = self.og_user_effect_sizes[1]
                else:
                    self.user_effect_sizes = self.user_effect_sizes / self.delayed_effect_scale_val
                self.times_shrunk -= 1


    def get_states(self):
        return self.user_states

    def get_user_effect_sizes(self):
        return self.user_effect_sizes

"""## SIMULATING DELAYED EFFECTS COMPONENT
---
"""
def create_user_envs(users_list, effect_size_scale, delayed_effect_scale_val, env_type):
    all_user_envs = {}
    for i, user in enumerate(users_list):
      new_user = UserEnvironment(user, effect_size_scale, env_type, delayed_effect_scale_val)
      all_user_envs[i] = new_user

    return all_user_envs

def get_delayed_effect_scale(delayed_effect_scale):
    if delayed_effect_scale == 'LOW_R':
        return 0
    elif delayed_effect_scale == 'MED_R':
        return 0.5
    elif delayed_effect_scale == 'HIGH_R':
        return 0.8
    else:
        print("ERROR: NO DELAYED EFFECT SCALE FOUND - ", delayed_effect_scale)


class SimulationEnvironmentV1(simulation_environment.SimulationEnvironment):
    def __init__(self, users_list, env_type, effect_size_scale, delayed_effect_scale):
        process_env_state_func = lambda session, j, Qs: process_env_state(session, j, Qs, env_type)
        delayed_effect_scale_val = get_delayed_effect_scale(delayed_effect_scale)
        user_envs = create_user_envs(users_list, effect_size_scale, delayed_effect_scale_val, env_type)

        super(SimulationEnvironmentV1, self).__init__(users_list, process_env_state_func, user_envs)

        self.env_type = env_type
        # Dimension of the environment state space
        self.dimension = 5 if env_type == 'STAT' else 6

    def update_responsiveness(self, user_idx, a1_cond, a2_cond, b_cond, j):
        self.all_user_envs[user_idx].update_responsiveness(a1_cond, a2_cond, b_cond, j)
