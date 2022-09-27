# -*- coding: utf-8 -*-
"""
Calculates desired statistics
1. Policy Parameter Estimating Equation
2. Policy Derivatives
"""

import numpy as np
from rl_algorithm import *

"""
Helpers
"""
def V(Sigma):
    n = len(Sigma)

    return 1/n * np.linalg.inv(Sigma)

"""
1. Policy Parameter Estimating Equation
"""
# user history is a tuple where tuple[0] is a matrix of stacked phi's specific to the user
# and tuple[1] is a vector of stacked rewards specific to the user
def compute_estimating_equation(user_history, mu, Sigma, prior_mu, prior_Sigma, sigma_n_squared):
    V = V(Sigma)
    n = len(Sigma)
    user_phi = user_history[0]
    R = user_history[1]

    mean_estimate = (1/sigma_n_squared) * G.T @ (R - G @ mu) \
    - (1/n) * np.linalg.inv(prior_Sigma) @ (mu - prior_mu)
    var_estimate = (1/sigma_n_squared) * G.T @ G + (1/n) *  np.linalg.inv(prior_Sigma) - V

    return mean_estimate, var_estimate

"""
2. Policy Derivatives
"""
