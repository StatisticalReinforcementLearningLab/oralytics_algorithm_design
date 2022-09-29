# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

ROBAS_3_STAT_PARAMS_DF = pd.read_csv('../sim_env_data/stat_user_models.csv')
ROBAS_3_NON_STAT_PARAMS_DF = pd.read_csv('../sim_env_data/non_stat_user_models.csv')
robas_3_data_df = pd.read_csv("https://raw.githubusercontent.com/ROBAS-UCLA/ROBAS.3/main/data/robas_3_data.csv")
ROBAS_3_USERS = np.array(ROBAS_3_STAT_PARAMS_DF['User'])
NUM_USERS = len(ROBAS_3_USERS)

### NORMALIZTIONS ###
def normalize_total_brush_quality(quality):
  return (quality - 154) / 163

def normalize_day_in_study(day):
  return (day - 35.5) / 34.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

### STATE SPACE ###
def get_user_df(user_id):
  return robas_3_data_df[robas_3_data_df['ROBAS ID'] == user_id]

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
  first_weekend_idx = np.where(np.array(user_df['Day Type']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,3] = 1

  return states

def generate_state_spaces_non_stat(user_df, num_days):
  ## init ##
  D = 6
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # day in study
    states[i][2] = normalize_day_in_study(i // 2 + 1)
    # bias term
    states[i][5] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['Day Type']) == 1)[0][0]
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

def process_env_state(session, j, Qs, env_type='stat'):
    env_state = session.copy()
    # session type - either 0 or 1
    session_type = int(env_state[0])
    # update previous day total brush time
    previous_day_total_rewards = get_previous_day_total_brush_quality(Qs, session[0], j)
    env_state[1] = normalize_total_brush_quality(previous_day_total_rewards)
    # proportion of past success brushing
    prior_idx = 2 if env_type == 'stat' else 3
    if (j >= 14):
      env_state[prior_idx] = np.sum([Qs[j - k] > 0.0 for k in range(1, 15)]) / 14

    return env_state

"""## Generate States
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

"""### Functions for Environment Models
---
"""
def construct_model_and_sample(user, state, action, \
                                          bern_params, \
                                          y_params, \
                                          sigma_u, \
                                          model_type, \
                                          effect_func_bern=lambda state : 0, \
                                          effect_func_y=lambda state : 0):
  bern_linear_comp = state @ bern_params
  if (action == 1):
    bern_linear_comp += effect_func_bern(state)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
      y_mu = state @ y_params
      if (action == 1):
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

"""## Constructing the Effect Sizes
---
Effect size are imputed using the each environment's (stationary/non-stationary) fitted parameters
"""

bern_param_titles = ['Time.of.Day.Bern', \
                     'Prior.Day.Total.Brush.Time.norm.Bern', \
                     'Proportion.Brushed.In.Past.7.Days.Bern', \
                     'Day.Type.Bern']

y_param_titles = ['Time.of.Day.Y', \
                        'Prior.Day.Total.Brush.Time.norm.Y', \
                     'Proportion.Brushed.In.Past.7.Days.Y', \
                     'Day.Type.Y']


# returns dictionary of effect sizes where the key is the user_id
# and the value is a tuple where the tuple[0] is the bernoulli effect size
# and tuple[1] is the effect size on y
# users are grouped by their base model type first when calculating imputed effect sizes
def get_effect_sizes(parameter_df):
    hurdle_df = parameter_df[parameter_df['Model Type'] == 'sqrt_norm']
    zip_df = parameter_df[parameter_df['Model Type'] == 'zero_infl']
    ### HURDLE ###
    # effect size mean
    hurdle_bern_mean = np.mean(np.mean(np.abs(np.array([hurdle_df[title] for title in bern_param_titles])), axis=1))
    hurdle_y_mean = np.mean(np.mean(np.abs(np.array([hurdle_df[title] for title in y_param_titles])), axis=1))
    # effect size std
    hurdle_bern_std = np.std(np.mean(np.abs(np.array([hurdle_df[title] for title in bern_param_titles])), axis=0))
    hurdle_y_std = np.std(np.mean(np.abs(np.array([hurdle_df[title] for title in y_param_titles])), axis=0))

    ### ZIP ###
    # effect size mean
    zip_bern_mean = np.mean(np.mean(np.abs(np.array([zip_df[title] for title in bern_param_titles])), axis=1))
    zip_y_mean = np.mean(np.mean(np.abs(np.array([zip_df[title] for title in y_param_titles])), axis=1))
    # effect size std
    zip_bern_std = np.std(np.mean(np.abs(np.array([zip_df[title] for title in bern_param_titles])), axis=0))
    zip_y_std = np.std(np.mean(np.abs(np.array([zip_df[title] for title in y_param_titles])), axis=0))

    ## simulating the effect sizes per user ##
    user_effect_sizes = {}
    np.random.seed(1)
    for user in parameter_df['User']:
        if np.array(parameter_df[parameter_df['User'] == user])[0][2] == "sqrt_norm":
            bern_eff_size = np.random.normal(loc=hurdle_bern_mean, scale=hurdle_bern_std)
            y_eff_size = np.random.normal(loc=hurdle_y_mean, scale=hurdle_y_std)
        else:
            bern_eff_size = np.random.normal(loc=zip_bern_mean, scale=zip_bern_std)
            y_eff_size = np.random.normal(loc=zip_y_mean, scale=zip_y_std)

        user_effect_sizes[user] = [bern_eff_size, y_eff_size]

    return user_effect_sizes

STAT_USER_EFFECT_SIZES = get_effect_sizes(ROBAS_3_STAT_PARAMS_DF)
NON_STAT_USER_EFFECT_SIZES = get_effect_sizes(ROBAS_3_NON_STAT_PARAMS_DF)

print("STAT USER EFFECT SIZES: ", STAT_USER_EFFECT_SIZES)
print("NON STAT USER EFFECT SIZES: ", NON_STAT_USER_EFFECT_SIZES)

## USER-SPECIFIC EFFECT SIZES ##
# Context-Aware with all features same as baseline features excpet for Prop. Non-Zero Brushing In Past 7 Days
# which is of index 2 for stat models and of index 3 for non stat models
stat_user_spec_effect_func_bern = lambda state, effect_size: -1.0*max(np.array(4 * [effect_size]) @ np.delete(state, 2), 0)
stat_user_spec_effect_func_y = lambda state, effect_size: max(np.array(4 * [effect_size]) @ np.delete(state, 2), 0)
non_stat_user_spec_effect_func_bern = lambda state, effect_size: -1.0*max(np.array(5 * [effect_size]) @ np.delete(state, 3), 0)
non_stat_user_spec_effect_func_y = lambda state, effect_size: max(np.array(5 * [effect_size]) @ np.delete(state, 3), 0)


"""## Creating Simulation Environment Objects
---
"""

### ANNA TODO: think of an easier way to handle stationary vs. non-stationary ###
class UserEnvironment():
    def __init__(self, user_id, env_type, unresponsive_val):
        self.user_id = user_id
        self.model_type = np.array(ROBAS_3_STAT_PARAMS_DF[ROBAS_3_STAT_PARAMS_DF['User'] == user_id])[0][2] \
        if env_type == 'stat' else np.array(ROBAS_3_NON_STAT_PARAMS_DF[ROBAS_3_NON_STAT_PARAMS_DF['User'] == user])[0][2]
        # vector: size (T, D) where D = 6 is the dimension of the env. state
        # T is the length of the study
        self.user_states = USERS_SESSIONS_STAT[user_id] if env_type == 'stat' else USERS_SESSIONS_NON_STAT[user_id]
        # tuple: float values of effect size on bernoulli, poisson components
        self.og_user_effect_sizes = np.array(STAT_USER_EFFECT_SIZES[user_id]) if env_type == 'stat' else np.array(NON_STAT_USER_EFFECT_SIZES[user_id])
        self.user_effect_sizes = np.copy(self.og_user_effect_sizes)
        # float: unresponsive scaling value
        self.unresponsive_val = unresponsive_val
        # probability of becoming unresponsive
        ### SETTING TO 1 FOR NOW!
        self.unresponsive_prob = 1.0
        # you can only shrink at most once a week
        self.times_shrunk = 0
        # reward generating function
        self.user_params = get_params_for_user(user_id, env_type)
        self.user_effect_func_bern = stat_user_spec_effect_func_bern if env_type == 'stat' else non_stat_user_spec_effect_func_bern
        self.user_effect_func_y = stat_user_spec_effect_func_y if env_type == 'stat' else non_stat_user_spec_effect_func_y
        self.reward_generating_func = lambda state, action: construct_model_and_sample(user_id, state, action, \
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
                    if (b_cond and a1_cond) and a2_cond == False:
                        DEBUG_NUM_EFFECT_SHRINKS_1 += 1
                    elif a2_cond and ((b_cond and a1_cond) == False):
                        DEBUG_NUM_EFFECT_SHRINKS_2 += 1
                    else:
                        DEBUG_NUM_EFFECT_SHRINKS_3 += 1

                    self.user_effect_sizes = self.user_effect_sizes * self.unresponsive_val
                    self.times_shrunk += 1

            elif self.times_shrunk > 0:
                if self.unresponsive_val == 0:
                    self.user_effect_sizes[0] = self.og_user_effect_sizes[0]
                    self.user_effect_sizes[1] = self.og_user_effect_sizes[1]
                else:
                    self.user_effect_sizes = self.user_effect_sizes / self.unresponsive_val
                self.times_shrunk -= 1


    def get_states(self):
        return self.user_states

    def get_user_effect_sizes(self):
        return self.user_effect_sizes

### SIMULATE DELAYED EFFECTS ###
def create_user_envs(users_list, unresponsive_val, env_type):
    all_user_envs = {}
    for i, user in enumerate(users_list):
      new_user = UserEnvironment(user, env_type, unresponsive_val)
      all_user_envs[i] = new_user

    return all_user_envs

class SimulationEnvironment():
    def __init__(self, users_list, env_type, unresponsive_val):
        # Func
        self.process_env_state = lambda session, j, Qs: process_env_state(session, j, Qs, env_type)
        # Dict: key: String user_id, val: user environment object
        self.all_user_envs = create_user_envs(users_list, unresponsive_val, env_type)
        # List: users in the environment (can repeat)
        self.users_list = users_list

    def generate_rewards(self, user_idx, state, action):
        return self.all_user_envs[user_idx].generate_reward(state, action)

    def get_states_for_user(self, user_idx):
        return self.all_user_envs[user_idx].get_states()

    def get_users(self):
        return self.users_list

    def update_responsiveness(self, user_idx, a1_cond, a2_cond, b_cond, j):
        self.all_user_envs[user_idx].update_responsiveness(a1_cond, a2_cond, b_cond, j)

### SIMULATION ENV AXIS VALUES ###
# These are the values you can tweak for the variants of the simulation environment
RESPONSIVITY_SCALING_VALS = [0, 0.5, 0.8]

STAT_LOW_R = lambda users_list: SimulationEnvironment(users_list, 'stat', RESPONSIVITY_SCALING_VALS[0])
STAT_MED_R = lambda users_list: SimulationEnvironment(users_list, 'stat', RESPONSIVITY_SCALING_VALS[1])
STAT_HIGH_R = lambda users_list: SimulationEnvironment(users_list, 'stat', RESPONSIVITY_SCALING_VALS[2])
NON_STAT_LOW_R = lambda users_list: SimulationEnvironment(users_list, 'non_stat', RESPONSIVITY_SCALING_VALS[0])
NON_STAT_MED_R = lambda users_list: SimulationEnvironment(users_list, 'non_stat', RESPONSIVITY_SCALING_VALS[1])
NON_STAT_HIGH_R = lambda users_list: SimulationEnvironment(users_list, 'non_stat', RESPONSIVITY_SCALING_VALS[2])
