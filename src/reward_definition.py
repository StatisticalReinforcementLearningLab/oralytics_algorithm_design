# -*- coding: utf-8 -*-
"""
RL Algorithm that uses a contextual bandit framework with Thompson sampling, full-pooling, and
a Bayesian Linear Regression reward approximating function.
"""
import numpy as np

### Reward Definition ###
GAMMA = 13/14
B = 111
A_1 = 0.5
A_2 = 0.8
DISCOUNTED_GAMMA_ARRAY = GAMMA ** np.flip(np.arange(14))
CONSTANT = (1 - GAMMA) / (1 - GAMMA**14)

# given user_qualities and user_actions, return b_bar and a_bar
# user_qualities and user_actions should be numpy arrays of the same size
def get_b_bar_a_bar(user_qualities, user_actions):
    j = len(user_actions)
    if j < 14:
        a_bar = 0 if len(user_actions) == 0 else np.mean(user_actions)
        b_bar = 0 if len(user_qualities) == 0 else np.mean(user_qualities)
    else:
        a_bar = calculate_a_bar(user_actions[-14:])
        b_bar = calculate_b_bar(user_qualities[-14:])

    return b_bar, a_bar

# b bar is designed to be in [0, 180]
# we want normalized b_bar to be close to [-1, 1]
def normalize_b_bar(b_bar):
  return (b_bar - (181 / 2)) / (179 / 2)

# we need a_bar to be between -1 and 1 because for behavioral scientists to interpret
# the intercept term, all state features need to have meaning at value 0.
# unnormalized a_bar is between [0, 1]
def normalize_a_bar(a_bar):
  return 2 * (a_bar - 0.5)

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
def calculate_reward(brushing_quality, xi_1, xi_2, current_action,\
                      b_bar, a_bar):
  B_condition = calculate_b_condition(b_bar)
  A1_condition = calculate_a1_condition(a_bar)
  A2_condition = calculate_a2_condition(a_bar)
  C = cost_definition(xi_1, xi_2, current_action, B_condition, A1_condition, A2_condition)

  return brushing_quality - C
