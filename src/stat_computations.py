# -*- coding: utf-8 -*-
"""
Calculates desired statistics
1. Policy Parameter Estimating Equation
2. Policy Derivatives
"""

import numpy as np

"""
Helpers
"""
# n is the number of users currently in the study
def scaled_inv_sigma(Sigma, n):

    return 1/n * np.linalg.inv(Sigma)

"""
1. Policy Parameter Estimating Equation
"""
# user history is a tuple where tuple[0] is a matrix of stacked phi's specific to the user
# and tuple[1] is a vector of stacked rewards specific to the user
# n is the number of users currently in the study
def compute_estimating_equation(user_history, n, mu, Sigma, prior_mu, prior_Sigma, sigma_n_squared):
    V = scaled_inv_sigma(Sigma, n)
    Phi = user_history[0]
    R = user_history[1]

    mean_estimate = (1/sigma_n_squared) * Phi.T @ (R - Phi @ mu) \
    - (1/n) * np.linalg.inv(prior_Sigma) @ (mu - prior_mu)
    var_estimate = (1/sigma_n_squared) * Phi.T @ Phi + (1/n) *  np.linalg.inv(prior_Sigma) - V

    return np.concatenate([mean_estimate, var_estimate.flatten()])

"""
2. Policy Derivatives
"""
