# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm

import read_write_info

class SimulationEnvironment():
    def __init__(self, users_list, user_envs, env_type):
        # must be implemented by children
        self.version = None
        # List: users in the environment (can repeat)
        self.users_list = users_list
        # Dict: key: int trial_user_idx, val: user environment object
        self.all_user_envs = user_envs
        # STAT or NON_STAT
        self.env_type = env_type

    def get_version(self):
        return self.version

    def get_env_type(self):
        return self.env_type

    # this method needs to be implemented by all children
    def generate_current_state(self):
        return None

    # this method needs to be implemented by all children
    def get_env_history(self):
        return None

    # this method needs to be implemented by all children
    def set_env_history(self):
        return None

    def generate_rewards(self, user_idx, state, action):
        return self.all_user_envs[user_idx].generate_reward(state, action)

    def get_states_for_user(self, user_idx):
        return self.all_user_envs[user_idx].get_states()

    def get_users(self):
        return self.users_list

    def get_env_history(self, user_idx, property):
        return self.all_user_envs[user_idx].get_user_history(property)

    def set_env_history(self, user_idx, property, value):
        self.all_user_envs[user_idx].set_user_history(property, value)

    def update_responsiveness(self, user_idx, a1_cond, a2_cond, b_cond, j):
        self.all_user_envs[user_idx].update_responsiveness(a1_cond, a2_cond, b_cond, j)

class SimulationEnvironmentAppEngagement(SimulationEnvironment):
    def __init__(self, users_list, user_envs, env_type):
        super(SimulationEnvironmentAppEngagement, self).__init__(users_list, user_envs, env_type)

    def generate_app_engagement(self, user_idx):
        return self.all_user_envs[user_idx].generate_app_engagement()

    def get_user_prior_day_app_engagement(self, user_idx):
        return self.all_user_envs[user_idx].get_prior_day_app_engagement()

    def set_user_prior_day_app_engagement(self, user_idx, prior_app_engagement):
        self.all_user_envs[user_idx].set_prior_day_app_engagement(prior_app_engagement)

    def get_user_last_open_app_dt(self, user_idx):
        return self.all_user_envs[user_idx].get_last_open_app_dt()

    def set_user_last_open_app_dt(self, user_idx, j):
        self.all_user_envs[user_idx].set_last_open_app_dt(j)

    def simulate_app_opening_behavior(self, user_idx, j):
        # we simulate that we only know if users opened their app in the morning
        if j % 2 == 0:
            # simulate whether or not the user opened their app
            current_app_engagement = self.generate_app_engagement(user_idx)
            if current_app_engagement:
                self.set_user_last_open_app_dt(user_idx, j)
        # we do not save that current day's app engagement until after the evening dt
        else:
            current_app_engagement = int(self.get_user_last_open_app_dt(user_idx) == j - 1)
            self.set_user_prior_day_app_engagement(user_idx, current_app_engagement)


# ### NORMALIZTIONS ###
def normalize_total_brush_quality(quality):
  return (quality - 154) / 163

def normalize_day_in_study(day):
  return (day - 35.5) / 34.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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

class UserEnvironment():
    def __init__(self, user_id, model_type, user_sessions, user_effect_sizes, \
                delayed_effect_scale_val, user_params, user_effect_func_bern, user_effect_func_y):
        self.user_id = user_id
        self.model_type = model_type
        # vector: size (T, D) where D is the dimension of the env. state
        # T is the length of the study
        self.user_states = user_sessions
        # tuple: float values of effect size on bernoulli, poisson components
        self.og_user_effect_sizes = user_effect_sizes
        self.user_effect_sizes = np.copy(self.og_user_effect_sizes)
        # float: unresponsive scaling value
        self.delayed_effect_scale_val = delayed_effect_scale_val
        # probability of becoming unresponsive
        ### SETTING TO 1 FOR NOW!
        self.unresponsive_prob = 1.0
        # you can only shrink at most once a week
        self.times_shrunk = 0
        # reward generating function
        self.user_params = user_params
        self.user_effect_func_bern = user_effect_func_bern
        self.user_effect_func_y = user_effect_func_y
        self.reward_generating_func = lambda state, action: construct_model_and_sample(user_id, state, action, \
                                          self.user_params[0], \
                                          self.user_params[1], \
                                          self.user_params[2], \
                                          self.model_type, \
                                          effect_func_bern=lambda state: self.user_effect_func_bern(state, self.user_effect_sizes[0]), \
                                          effect_func_y=lambda state: self.user_effect_func_y(state, self.user_effect_sizes[1]))
        # user environment history
        self.user_history = {"actions":[], "outcomes":[]}

    def get_user_history(self, property):
        return self.user_history[property]

    def set_user_history(self, property, value):
        self.user_history[property].append(value)

    def generate_reward(self, state, action):
        # save action and outcome
        self.set_user_history("actions", action)
        reward = min(self.reward_generating_func(state, action), 180)
        self.set_user_history("outcomes", reward)

        return reward

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

class UserEnvironmentAppEngagement(UserEnvironment):
    def __init__(self, user_id, model_type, user_effect_sizes, delayed_effect_scale_val, \
                user_params, user_effect_func_bern, user_effect_func_y):
        super(UserEnvironmentAppEngagement, self).__init__(user_id, model_type, None, user_effect_sizes, \
                  delayed_effect_scale_val, user_params, user_effect_func_bern, user_effect_func_y)
        # probability of opening app, needs to be implemented by children
        self.app_open_base_prob = None
        # tracking prior day app engagement
        self.prior_day_app_engagement = 0
        # last dt user open app, init: all users are assumed to open their app on the first day
        self.last_open_app_dt = 0

    def generate_app_engagement(self):
        return bernoulli.rvs(self.app_open_base_prob)

    def get_prior_day_app_engagement(self):
        return self.prior_day_app_engagement

    def set_prior_day_app_engagement(self, prior_day_app_engage):
        self.prior_day_app_engagement = prior_day_app_engage

    def get_last_open_app_dt(self):
        return self.last_open_app_dt

    def set_last_open_app_dt(self, j):
        self.last_open_app_dt = j

"""## SIMULATING DELAYED EFFECTS COMPONENT
---
"""

def get_delayed_effect_scale(delayed_effect_scale):
    if delayed_effect_scale == 'LOW_R':
        return 0
    elif delayed_effect_scale == 'MED_R':
        return 0.5
    elif delayed_effect_scale == 'HIGH_R':
        return 0.8
    else:
        print("ERROR: NO DELAYED EFFECT SCALE FOUND - ", delayed_effect_scale)
