import numpy as np
"""### Smoothing Functions
---
"""
# traditional Thompson Sampling
BASIC_THOMPSON_SAMPLING_FUNC = lambda x: x > 0

# generalized logistic function https://en.wikipedia.org/wiki/Generalised_logistic_function
# lower and upper asymptotes
L_min = 0.1
L_max = 0.9
# larger values of b > 0 makes curve more "steep"
B_logistic = 10
# larger values of c > 0 shifts the value of function(0) to the right
C_logistic = 3
# larger values of k > 0 makes the asmptote towards upper clipping less steep
# and the asymptote towards the lower clipping more steep
K_logistic = 1

# return the value we want for C_logistic
# rationale := at x = 0, we want the desired generalized logistic value
def calculate_C_logistic(L_min, L_max, desired_value_at_0):
    return ((L_max - L_min) / (desired_value_at_0 - L_min)) - 1

# return the value we want for B_logistic
# rationale := at x = 0.2, we want the desired generalized logistic value
def calculate_B_logistic(L_min, L_max, C_logistic, desidesired_value_at_0, desired_value_at_02):

    return 0 # ANNA TODO

# print("0.1, 0.9", calculate_C_logistic(0.1, 0.9, 1.2*0.1))
# print("0.35, 0.75", calculate_C_logistic(0.35, 0.75, 1.2*0.35))

def genearlized_logistic_func(x):
    num = L_max - L_min
    denom = (1 + C_logistic * np.exp(-B_logistic * x))**K_logistic

    return L_min + (num / denom)

GENERALIZED_LOGISTIC_FUNC = lambda x: np.apply_along_axis(genearlized_logistic_func, 0, x)
