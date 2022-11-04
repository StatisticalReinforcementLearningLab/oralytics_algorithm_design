import numpy as np
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
def genearlized_logistic_func(x, l_min, l_max, B):
    num = l_max - l_min
    denom = (1 + C_logistic * np.exp(-B * x))**K_logistic

    return l_min + (num / denom)

def genearlized_logistic_func_wrapper(l_min, l_max, B):
    smoothing_func = lambda x: genearlized_logistic_func(x, l_min, l_max, B)

    return lambda x: np.apply_along_axis(smoothing_func, 0, x)
