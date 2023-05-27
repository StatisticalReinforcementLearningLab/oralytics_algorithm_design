# -*- coding: utf-8 -*-
"""
RL Algorithm that uses a contextual bandit framework with Thompson sampling, full-pooling, and
a Bayesian Linear Regression reward approximating function.
"""
import stat_computations
import reward_definition

import pandas as pd
import numpy as np
import scipy.stats as stats

class RLAlgorithm():
    def __init__(self, cost_params, update_cadence, smoothing_func):
        # how often the RL algorithm updates parameters
        self.update_cadence = update_cadence
        # smoothing function for after-study analysis
        self.smoothing_func = smoothing_func
        # feature space dimension
        self.feature_dim = 0
        # xi_1, xi_2 params for the cost term parameterizes the reward def. func.
        self.reward_def_func = lambda brushing_quality, current_action, b_bar, a_bar: \
                      reward_definition.calculate_reward(brushing_quality, \
                                        cost_params[0], cost_params[1], \
                                        current_action, b_bar, a_bar)

    # need to implement
    # function that takes in a raw state and processes the current state for the algorithm
    def process_alg_state(self, env_state, b_bar, a_bar):
        return 0

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, alg_states, actions, pis, rewards):
        return 0

    def get_feature_dim(self):
        return self.feature_dim

    def get_update_cadence(self):
        return self.update_cadence

"""
Algorithm State Space (For V1)
"""
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
def process_alg_state_v1(env_state, b_bar, a_bar):
    baseline_state = np.array([env_state[0], reward_definition.normalize_b_bar(b_bar), \
                               a_bar, 1])
    advantage_state = np.copy(baseline_state)

    return advantage_state, baseline_state

"""
Algorithm State Space (For V2)
"""
# please refer to generate_env_state in sim_env_v2
## baseline: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - app engagement
# 4 - bias
## advantage: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - app engagement
# 4 - bias
def process_alg_state_v2(env_state, b_bar, a_bar):
    baseline_state = np.array([env_state[0], reward_definition.normalize_b_bar(b_bar), \
                               reward_definition.normalize_a_bar(a_bar), env_state[-1], 1])
    advantage_state = np.copy(baseline_state)

    return advantage_state, baseline_state

"""
Algorithm State Space (For V3)
"""
# please refer to generate_env_state in sim_env_v3
## baseline: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - app engagement
# 4 - bias
## advantage: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - app engagement
# 4 - bias
def process_alg_state_v3(env_state):
    baseline_state = np.array([env_state[0], env_state[1], \
                               env_state[2], env_state[3], 1])
    advantage_state = np.copy(baseline_state)

    return advantage_state, baseline_state

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

## ACTION SELECTION ##
# we calculate the posterior probability of P(R_1 > R_0) clipped
# we make a Bernoulli draw with prob. P(R_1 > R_0) of the action
def bayes_lr_action_selector(beta_post_mean, beta_post_var, advantage_state, smoothing_func):
  # using the genearlized_logistic_func, probabilities are already clipped to asymptotes
  mu = advantage_state @ beta_post_mean
  std = np.sqrt(advantage_state @ beta_post_var @ advantage_state.T)
  posterior_prob = stats.norm.expect(func=smoothing_func, loc=mu, scale=std)

  return stats.bernoulli.rvs(posterior_prob), posterior_prob

"""### BLR Algorithm Object
---
"""
class BayesianLinearRegression(RLAlgorithm):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BayesianLinearRegression, self).__init__(cost_params, update_cadence, smoothing_func)

        # need to be set by children classes
        self.D_ADVANTAGE = None
        self.D_BASELINE = None
        self.PRIOR_MU = None
        self.PRIOR_SIGMA = None
        self.SIGMA_N_2 = None
        self.feature_map = None
        self.posterior_mean = None
        self.posterior_var = None
        # parameter used to control study-level period of pure exploration
        self.use_prior = True

    def action_selection(self, advantage_state):
        if self.use_prior:
            return bayes_lr_action_selector(self.PRIOR_MU[-self.D_ADVANTAGE:], \
                                                self.PRIOR_SIGMA[-self.D_ADVANTAGE:,-self.D_ADVANTAGE:], \
                                                advantage_state, \
                                                self.smoothing_func)
        else:
            return bayes_lr_action_selector(self.posterior_mean[-self.D_ADVANTAGE:], \
                                                self.posterior_var[-self.D_ADVANTAGE:,-self.D_ADVANTAGE:], \
                                                advantage_state, \
                                                self.smoothing_func)
    def is_pure_exploration_period(self):
        return self.use_prior

    def end_pure_exploration_period(self):
        self.use_prior = False

    def update(self, alg_states, actions, pis, rewards):
        Phi = self.feature_map(alg_states, alg_states, actions, pis)
        posterior_mean, posterior_var = update_posterior_w(Phi, rewards, self.SIGMA_N_2, self.PRIOR_MU, self.PRIOR_SIGMA)
        self.posterior_mean = posterior_mean
        self.posterior_var = posterior_var

    def compute_estimating_equation(self, user_history, n):
        return stat_computations.compute_estimating_equation(user_history, n, \
        self.posterior_mean, self.posterior_var, self.PRIOR_MU, self.PRIOR_SIGMA, self.SIGMA_N_2)

class BlrActionCentering(BayesianLinearRegression):
    def __init__(self, cost_params, update_cadence, smoothing_func, noise_var):
        super(BlrActionCentering, self).__init__(cost_params, update_cadence, smoothing_func)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline + D_advantage + D_advantage
        self.D_ADVANTAGE = 4
        self.D_BASELINE = 4
        self.feature_dim = self.D_BASELINE + self.D_ADVANTAGE + self.D_ADVANTAGE
        self.PRIOR_MU = np.array([0, 4.925, 0, 82.209, 0, 0, 0, 0, 0, 0, 0, 0])
        sigma_beta = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, sigma_beta**2, 46.240**2, \
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2,\
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2]))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = noise_var
        # feature map
        self.feature_map = create_big_phi

    def process_alg_state(self, env_state, b_bar, a_bar):
        return process_alg_state_v1(env_state, b_bar, a_bar)

# algorithm candidates that run in the V2 environment
# uses a prior built on Oralytics pilot data
class BlrACV2(BlrActionCentering):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BlrACV2, self).__init__(cost_params, update_cadence, smoothing_func, None)

        # THESE VALUES WERE SET WITH ORALYTICS PILOT DATA
        # size of mu vector = D_baseline=5 + D_advantage=5 + D_advantage=5
        self.D_ADVANTAGE = 5
        self.D_BASELINE = 5
        self.feature_dim = self.D_BASELINE + self.D_ADVANTAGE + self.D_ADVANTAGE
        ALPHA_0_MU = [0, 0, 17, 0, 77]
        BETA_MU = [0, 0, 0, 27, 0]
        ALPHA_0_SIGMA = [23**2, 12**2, 15**2, 8**2, 44**2]
        BETA_SIGMA = [12**2, 32**2, 29**2, 20**2, 13**2]
        self.PRIOR_MU = np.array(ALPHA_0_MU + BETA_MU + BETA_MU)
        self.PRIOR_SIGMA = np.diag(np.array(ALPHA_0_SIGMA + BETA_SIGMA + BETA_SIGMA))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = 4058

    def process_alg_state(self, env_state, b_bar, a_bar):
        return process_alg_state_v2(env_state, b_bar, a_bar)

# algorithm candidates that run in the V3 environment
# uses a prior built on ROBAS 2 data; this was the prior used in the Oralytics pilot data
class BlrACV3(BlrActionCentering):
    def __init__(self, cost_params, update_cadence, smoothing_func):
        super(BlrACV3, self).__init__(cost_params, update_cadence, smoothing_func, None)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline=5 + D_advantage=5 + D_advantage=5
        self.D_ADVANTAGE = 5
        self.D_BASELINE = 5
        self.feature_dim = self.D_BASELINE + self.D_ADVANTAGE + self.D_ADVANTAGE
        self.PRIOR_MU = np.array([0, 4.925, 0, 0, 82.209, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        SIGMA_BETA = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, SIGMA_BETA**2, SIGMA_BETA**2, 46.240**2, \
                                    SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2,\
                                    SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2, SIGMA_BETA**2]))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = 3396.449

    # Note: In V3, the environment state already calculates a normalized version of b_bar and a_bar
    def process_alg_state(self, env_state, b_bar, a_bar):
        return process_alg_state_v3(env_state)

class BlrNoActionCentering(BayesianLinearRegression):
    def __init__(self, cost_params, update_cadence, smoothing_func, noise_var):
        super(BlrNoActionCentering, self).__init__(cost_params, update_cadence, smoothing_func)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline + D_advantage
        self.D_ADVANTAGE = 4
        self.D_BASELINE = 4
        self.feature_dim = self.D_BASELINE + self.D_ADVANTAGE
        self.PRIOR_MU = np.array([0, 4.925, 0, 82.209, 0, 0, 0, 0])
        sigma_beta = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, sigma_beta**2, 46.240**2, \
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2]))
        self.posterior_mean = np.copy(self.PRIOR_MU)
        self.posterior_var = np.copy(self.PRIOR_SIGMA)

        self.SIGMA_N_2 = noise_var
        # feature map
        self.feature_map = lambda adv_states, base_states, probs, actions: \
        create_big_phi_no_action_centering(adv_states, base_states, actions)

    def process_alg_state(self, env_state, b_bar, a_bar):
        return process_alg_state_v1(env_state, b_bar, a_bar)
