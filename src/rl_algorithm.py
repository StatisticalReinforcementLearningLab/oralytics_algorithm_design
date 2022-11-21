# -*- coding: utf-8 -*-
"""
RL Algorithm that uses a contextual bandit framework with Thompson sampling, full-pooling, and
a Bayesian Linear Regression reward approximating function.
"""
# from src import stat_computations
import stat_computations

import pandas as pd
import numpy as np
from scipy.stats import bernoulli

class RLAlgorithm():
    def __init__(self, update_cadence, smoothing_func, process_alg_state_func):
        # how often the RL algorithm updates parameters
        self.update_cadence = update_cadence
        # smoothing function for after-study analysis
        self.smoothing_func = smoothing_func
        # function that takes in a raw state and processes the current state for the algorithm
        self.process_alg_state_func = process_alg_state_func
        # feature space dimension
        self.feature_dim = 0

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, alg_states, actions, pis, rewards):
        return 0

    def get_update_cadence(self):
        return self.update_cadence

# Advantage Time Feature Dimensions
D_advantage = 4
# Baseline Time Feature Dimensions
D_baseline = 4
# Number of Posterior Draws
NUM_POSTERIOR_SAMPLES = 5000

### Reward Definition ###
GAMMA = 13/14
B = 111
A_1 = 0.5
A_2 = 0.8
DISCOUNTED_GAMMA_ARRAY = GAMMA ** np.flip(np.arange(14))
CONSTANT = (1 - GAMMA) / (1 - GAMMA**14)

# b bar is designed to be in [0, 180]
def normalize_b_bar(b_bar):
  return (b_bar - (181 / 2)) / (179 / 2)

# brushing duration is of length 14 where the first element is the brushing duration
# at time t - 14 and the last element the brushing duration at time t - 1
def calculate_b_bar(brushing_durations):
  sum_term = DISCOUNTED_GAMMA_ARRAY * brushing_durations

  return CONSTANT * np.sum(sum_term)

def calculate_a_bar(past_actions):
  sum_term = DISCOUNTED_GAMMA_ARRAY * past_actions

  return CONSTANT * np.sum(sum_term)

def calculate_b_condition(b_bar):
  return b_bar > B

def calculate_a1_condition(a_bar):
  return a_bar > A_1

def calculate_a2_condition(a_bar):
  return a_bar > A_2

def cost_definition(xi_1, xi_2, action, B_condition, A1_condition, A2_condition):
  return action * (xi_1 * B_condition * A1_condition + xi_2 * A2_condition)

# returns the reward where the cost term is parameterized by xi_1, xi_2
def reward_definition(brushing_quality, xi_1, xi_2, current_action,\
                      b_bar, a_bar):
  B_condition = calculate_b_condition(b_bar)
  A1_condition = calculate_a1_condition(a_bar)
  A2_condition = calculate_a2_condition(a_bar)
  C = cost_definition(xi_1, xi_2, current_action, B_condition, A1_condition, A2_condition)

  return brushing_quality - C

## baseline: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - bias
## advantage: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - bias

def process_alg_state(env_state, b_bar, a_bar):
    baseline_state = np.array([env_state[0], normalize_b_bar(b_bar), \
                               calculate_a_bar(a_bar), 1])
    advantage_state = np.copy(baseline_state)

    return advantage_state, baseline_state

class RLAlgorithmExperimentCandidate(RLAlgorithm):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        # process_alg_state is a global function
        super(RLAlgorithmExperimentCandidate, self).__init__(update_cadence, smoothing_func, process_alg_state)
        # xi_1, xi_2 params for the cost term parameterizes the reward def. func.
        self.reward_def_func = lambda brushing_quality, current_action, b_bar, a_bar: \
                      reward_definition(brushing_quality, \
                                        cost_params[0], cost_params[1], \
                                        current_action, b_bar, a_bar)

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        return 0

    def get_update_cadence(self):
        return self.update_cadence

"""### Bayesian Linear Regression Thompson Sampler
---
"""
## POSTERIOR HELPERS ##
# create the feature vector given state, action, and action selection probability

# with action centering
def create_big_phi(advantage_states, baseline_states, actions, probs):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, probs).T, \
                       np.multiply(advantage_states.T, (actions - probs)).T,))
  return big_phi

# without action centering
def create_big_phi_no_action_centering(advantage_states, baseline_states, actions):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, actions).T))

  return big_phi

"""
#### Helper Functions
---
"""

def compute_posterior_var(Phi, sigma_n_squared, prior_sigma):
  return np.linalg.inv(1/sigma_n_squared * Phi.T @ Phi + np.linalg.inv(prior_sigma))

def compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma):

  return compute_posterior_var(Phi, sigma_n_squared, prior_sigma) \
   @ (1/sigma_n_squared * Phi.T @ R + np.linalg.inv(prior_sigma) @ prior_mu)

# update posterior distribution
def update_posterior_w(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  mean = compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma)
  var = compute_posterior_var(Phi, sigma_n_squared, prior_sigma)

  return mean, var

def get_beta_posterior_draws(posterior_mean, posterior_var):
  # grab last D_advantage of mean vector
  beta_post_mean = posterior_mean[-D_advantage:]
  # grab right bottom corner D_advantage x D_advantage submatrix
  beta_post_var = posterior_var[-D_advantage:,-D_advantage:]

  return np.random.multivariate_normal(beta_post_mean, beta_post_var, NUM_POSTERIOR_SAMPLES)


## ACTION SELECTION ##
# we calculate the posterior probability of P(R_1 > R_0) clipped
# we make a Bernoulli draw with prob. P(R_1 > R_0) of the action
def bayes_lr_action_selector(beta_posterior_draws, advantage_state, smoothing_func):
  # Note: the smoothing function inherently clips between L_min and L_max
  smooth_posterior_prob = np.mean(smoothing_func(beta_posterior_draws @ advantage_state))

  return bernoulli.rvs(smooth_posterior_prob), smooth_posterior_prob

"""### BLR Algorithm Object
---
"""
class BayesianLinearRegression(RLAlgorithmExperimentCandidate):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BayesianLinearRegression, self).__init__(cost_params, update_cadence, smoothing_func)

        # need to be set by children classes
        self.PRIOR_MU = None
        self.PRIOR_SIGMA = None
        self.SIGMA_N_2 = None
        self.feature_map = None
        self.posterior_mean = None
        self.posterior_var = None
        self.beta_posterior_draws = None

    def action_selection(self, advantage_state):
        return bayes_lr_action_selector(self.beta_posterior_draws, advantage_state, self.smoothing_func)

    def update(self, alg_states, actions, pis, rewards):
        Phi = self.feature_map(alg_states, alg_states, actions, pis)
        posterior_mean, posterior_var = update_posterior_w(Phi, rewards, self.SIGMA_N_2, self.PRIOR_MU, self.PRIOR_SIGMA)
        self.posterior_mean = posterior_mean
        self.posterior_var = posterior_var
        self.beta_posterior_draws = get_beta_posterior_draws(posterior_mean, posterior_var)

    def compute_estimating_equation(self, user_history, n):
        return stat_computations.compute_estimating_equation(user_history, n, \
        self.posterior_mean, self.posterior_var, self.PRIOR_MU, self.PRIOR_SIGMA, self.SIGMA_N_2)

class BlrActionCentering(BayesianLinearRegression):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BlrActionCentering, self).__init__(cost_params, update_cadence, smoothing_func)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline + D_advantage + D_advantage
        self.feature_dim = D_baseline + D_advantage + D_advantage
        self.PRIOR_MU = np.array([0, 4.925, 0, 82.209, 0, 0, 0, 0, 0, 0, 0, 0])
        sigma_beta = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, sigma_beta**2, 46.240**2, \
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2,\
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2]))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = 3396.449
        # initial draws are from the prior
        self.beta_posterior_draws = get_beta_posterior_draws(self.PRIOR_MU, self.PRIOR_SIGMA)
        # feature map
        self.feature_map = create_big_phi

class BlrNoActionCentering(BayesianLinearRegression):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BlrNoActionCentering, self).__init__(cost_params, update_cadence, smoothing_func)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline + D_advantage
        self.feature_dim = D_baseline + D_advantage
        self.PRIOR_MU = np.array([0, 4.925, 0, 82.209, 0, 0, 0, 0])
        sigma_beta = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, sigma_beta**2, 46.240**2, \
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2]))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = 3396.449
        # initial draws are from the prior
        self.beta_posterior_draws = get_beta_posterior_draws(self.PRIOR_MU, self.PRIOR_SIGMA)
        # feature map
        self.feature_map = lambda adv_states, base_states, probs, actions: \
        create_big_phi_no_action_centering(adv_states, base_states, actions)
