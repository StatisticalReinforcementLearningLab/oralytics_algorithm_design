import numpy as np
from scipy.special import expit

"""### Smoothing Functions
---
"""
### traditional Thompson Sampling ###
BASIC_THOMPSON_SAMPLING_FUNC = lambda x: x > 0


### generalized logistic function ###
# https://en.wikipedia.org/wiki/Generalised_logistic_function
# larger values of b > 0 makes curve more "steep"
# B_logistic = 6
# larger values of c > 0 shifts the value of function(0) to the right
C_logistic = 3
# larger values of k > 0 makes the asmptote towards upper clipping less steep
# and the asymptote towards the lower clipping more steep
K_logistic = 1

# l_min, l_max are lower and upper asymptotes
# uses scipy.special.expit for numerical stability
def stable_generalized_logistic(x, L_min, L_max, B_logistic):
    num = L_max - L_min
    stable_exp = expit(B_logistic * x - np.log(C_logistic))
    stable_exp_k = stable_exp**K_logistic

    return L_min + num * stable_exp_k

def genearlized_logistic_func_wrapper(l_min, l_max, B):

    return lambda x: stable_generalized_logistic(x, l_min, l_max, B)
