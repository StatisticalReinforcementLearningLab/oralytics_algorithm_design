"""
GLOBAL VALUES
"""

MAX_SEED_VAL = 100
NUM_TRIALS = 70
TRIAL_LENGTH_IN_WEEKS = 10
# We should have NUM_USERS x NUM_DECISION_TIMES datapoints for each saved value or
# statistic at the end of the study
NUM_DECISION_TIMES = 70 * 2

"""
V1:
* recruitment rate of 4 per week
* no app engagement in algorithm state
"""
# NUM_TRIAL_USERS = 72
# RECRUITMENT_RATE = 4
# FILL_IN_COLS = ['policy_idx', 'action', 'prob', 'reward', 'quality', 'state.tod', 'state.b.bar',\
#  'state.a.bar', 'state.bias']

"""
V2 and V3:
* recruitment rate of 5 per 2 weeks
* app engagement in algorithm state
"""
NUM_TRIAL_USERS = 70
RECRUITMENT_RATE = 5
FILL_IN_COLS = ['policy_idx', 'action', 'prob', 'reward', 'quality'] + ['state.tod', 'state.b.bar',\
 'state.a.bar', 'state.app.engage', 'state.bias']

"""
Used for no pooling
"""
# RECRUITMENT_RATE = 72
