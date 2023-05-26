# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from scipy.stats import bernoulli

import read_write_info
import simulation_environment
import reward_definition

"""## BASE ENVIRONMENT COMPONENT
---
"""

STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v3_stat_zip_model_params.csv')
NON_STAT_PARAMS_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v3_non_stat_zip_model_params.csv')
APP_OPEN_PROB_DF = pd.read_csv(read_write_info.READ_PATH_PREFIX + 'sim_env_data/v3_app_open_prob.csv')
SIM_ENV_USERS = np.array(STAT_PARAMS_DF['User'])
# value used by run_experiments to draw with replacement
NUM_USER_MODELS = len(SIM_ENV_USERS)

# dictionary where key is index and value is user_id
USER_INDICES = {}
for i, user_id in enumerate(SIM_ENV_USERS):
    USER_INDICES[i] = user_id

"""### Generate States
---
"""

def get_previous_day_qualities_and_actions(j, Qs, As):
    if j > 1:
        if j % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As

# Stationary State Space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias

# Non-stationary state space
# 0 - time of day
# 1 - b_bar (normalized)
# 2 - a_bar (normalized)
# 3 - app engagement
# 4 - weekday vs. weekend
# 5 - bias
# 6 - day in study

# Note: We assume that all participants start the study on Monday (j = 0 denotes)
# Monday morning. Therefore the first weekend idx is j = 10 (Saturday morning)
def generate_env_state(j, user_qualities, user_actions, app_engagement, env_type):
    env_state = np.ones(7) if env_type == 'NON_STAT' else np.ones(6)
    # session type - either 0 or 1
    session_type = j % 2
    env_state[0] = session_type
    # b_bar, a_bar (normalized)
    Qs, As = get_previous_day_qualities_and_actions(j, user_qualities, user_actions)
    b_bar, a_bar = reward_definition.get_b_bar_a_bar(Qs, As)
    env_state[1] = reward_definition.normalize_b_bar(b_bar)
    env_state[2] = reward_definition.normalize_a_bar(a_bar)
    # app engagement
    env_state[3] = app_engagement
    # weekday vs. weekend
    env_state[4] = 1 if (j % 14 >= 10 and j % 14 <= 13) else 0
    # bias
    env_state[5] = 1
    # day in study if a non-stationary environment
    if env_type == 'NON_STAT':
        env_state[6] = simulation_environment.normalize_day_in_study(1 + (j // 2))

    return env_state

def get_app_open_prob(user_id):
    return APP_OPEN_PROB_DF[APP_OPEN_PROB_DF['user_id'] == user_id]['app_open_prob'].values[0]

# note: since v3 only chose zip models, these are the following parameters
def get_base_params_for_user(user, env_type='STAT'):
  param_dim = 6 if env_type == 'STAT' else 7
  param_df = STAT_PARAMS_DF if env_type=='STAT' else NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_base = user_row[0][2:2 + param_dim]
  poisson_base = user_row[0][2 + 2 * param_dim:2 + 3 * param_dim]

  # poisson parameters, bernouilli parameters
  return bern_base, poisson_base, None

# note: since v3 only chose zip models, these are the following parameters
def get_adv_params_for_user(user, env_type='STAT'):
  param_dim = 6 if env_type == 'STAT' else 7
  param_df = STAT_PARAMS_DF if env_type=='STAT' else NON_STAT_PARAMS_DF
  user_row = np.array(param_df[param_df['User'] == user])
  bern_adv = user_row[0][2 + param_dim: 2 + 2 * param_dim]
  poisson_adv = user_row[0][2 + 3 * param_dim:2 + 4 * param_dim]

  return bern_adv, poisson_adv

def get_user_effect_funcs():
    bern_adv_func = lambda state, adv_params: adv_params @ state
    y_adv_func = lambda state, adv_params: adv_params @ state

    return bern_adv_func, y_adv_func

"""## Creating Simulation Environment Objects
---
"""

class UserEnvironmentV3(simulation_environment.UserEnvironmentAppEngagement):
    def __init__(self, user_id, model_type, user_params, adv_params, \
                user_effect_func_bern, user_effect_func_y):
        # Note: in the base UserEnvironment, it uses simulated user_effect_sizes,
        # but we replace it with adv_params, user's fitted advantage parameters
        super(UserEnvironmentV3, self).__init__(user_id, model_type, adv_params, None, \
                user_params, user_effect_func_bern, user_effect_func_y)
        # probability of opening app
        self.app_open_base_prob = get_app_open_prob(user_id)

    # we no longer simulate delayed effects because pilot data had outcomes under action 1
    def update_responsiveness(self, a1_cond, a2_cond, b_cond, j):
        return None

def create_user_envs(users_list, env_type):
    all_user_envs = {}
    for i, user_id in enumerate(users_list):
      model_type = "zip" # note: all users in V3 have the zero-inflated poisson model
      base_params = get_base_params_for_user(user_id, env_type)
      adv_params = get_adv_params_for_user(user_id, env_type)
      user_effect_func_bern, user_effect_func_y = get_user_effect_funcs()
      new_user = UserEnvironmentV3(user_id, model_type, base_params, adv_params, \
                    user_effect_func_bern, user_effect_func_y)
      all_user_envs[i] = new_user

    return all_user_envs

class SimulationEnvironmentV3(simulation_environment.SimulationEnvironmentAppEngagement):
    # note: v3 does not have effect_size_scale, delayed_effect_scale properties
    def __init__(self, users_list, env_type, effect_size_scale, delayed_effect_scale):
        user_envs = create_user_envs(users_list, env_type)

        super(SimulationEnvironmentV3, self).__init__(users_list, user_envs, env_type)

        self.version = "V3"

    def generate_current_state(self, user_idx, j):
        # prior day app_engagement is 0 for the first day
        prior_app_engagement = self.get_user_prior_day_app_engagement(user_idx)
        self.simulate_app_opening_behavior(user_idx, j)
        brushing_qualities = np.array(self.get_env_history(user_idx, "outcomes"))
        past_actions = np.array(self.get_env_history(user_idx, "actions"))

        return generate_env_state(j, brushing_qualities, past_actions, prior_app_engagement, self.get_env_type())
