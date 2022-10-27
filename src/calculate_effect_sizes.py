# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle

class SimulationEnvironment():
    def __init__(self, users_list, process_env_state_func, user_envs):
        # List: users in the environment (can repeat)
        self.users_list = users_list
        # Func
        self.process_env_state = process_env_state_func
        # Dict: key: int trial_user_idx, val: user environment object
        self.all_user_envs = user_envs

    def generate_rewards(self, user_idx, state, action):
        return self.all_user_envs[user_idx].generate_reward(state, action)

    def get_states_for_user(self, user_idx):
        return self.all_user_envs[user_idx].get_states()

    def get_users(self):
        return self.users_list

ROBAS_3_STAT_PARAMS_DF = pd.read_csv('sim_env_data/stat_user_models.csv')
ROBAS_3_NON_STAT_PARAMS_DF = pd.read_csv('sim_env_data/non_stat_user_models.csv')

"""## Constructing the Effect Sizes
---
Effect sizes are imputed using the each environment's (stationary/non-stationary) fitted parameters
"""

BERN_PARAM_TITLES = ['Time.of.Day.Bern', \
                     'Prior.Day.Total.Brush.Time.norm.Bern', \
                     'Day.in.Study.norm.Bern', \
                     'Day.Type.Bern']

Y_PARAM_TITLES = ['Time.of.Day.Y', \
                  'Prior.Day.Total.Brush.Time.norm.Y', \
                  'Day.in.Study.norm.Y', \
                  'Day.Type.Y']


# returns dictionary of effect sizes where the key is the user_id
# and the value is a tuple where the tuple[0] is the bernoulli effect size
# and tuple[1] is the effect size on y
# users are grouped by their base model type first when calculating imputed effect sizes
def get_effect_sizes(parameter_df, bern_param_titles, y_param_titles, eny_type='stat'):
    hurdle_df = parameter_df[parameter_df['Model Type'] == 'sqrt_norm']
    zip_df = parameter_df[parameter_df['Model Type'] == 'zero_infl']
    shrinkage_value = 2
    get_mean_across_features = lambda array: np.mean(np.abs(array), axis=1) / shrinkage_value
    get_std_across_users = lambda array: np.std(np.abs(array) / shrinkage_value, axis=1)
    ### HURDLE ###
    hurdle_bern_param_array = np.array([hurdle_df[title] for title in bern_param_titles])
    hurdle_y_param_array = np.array([hurdle_df[title] for title in y_param_titles])
    # effect size bias mean
    hurdle_bern_mean_vector = np.concatenate([get_mean_across_features(hurdle_bern_param_array), \
    np.mean(get_mean_across_features(hurdle_bern_param_array))], axis=None)
    hurdle_y_mean_vector = np.concatenate([get_mean_across_features(hurdle_y_param_array), \
    np.mean(get_mean_across_features(hurdle_y_param_array))], axis=None)
    # effect size bias std
    hurdle_bern_std_diag = np.concatenate([get_std_across_users(hurdle_bern_param_array), \
    np.mean(get_std_across_users(hurdle_bern_param_array))], axis=None)
    hurdle_y_std_diag = np.concatenate([get_std_across_users(hurdle_y_param_array), \
    np.mean(get_std_across_users(hurdle_y_param_array))], axis=None)

    ### ZIP ###
    zip_bern_param_array = np.array([zip_df[title] for title in bern_param_titles])
    zip_y_param_array = np.array([zip_df[title] for title in y_param_titles])
    # effect size bias mean
    zip_bern_mean = np.concatenate([get_mean_across_features(zip_bern_param_array), \
    np.mean(get_mean_across_features(zip_bern_param_array))], axis=None)
    zip_y_mean = np.concatenate([get_mean_across_features(zip_y_param_array), \
    np.mean(get_mean_across_features(zip_y_param_array))], axis=None)
    # effect size bias std
    zip_bern_std = np.concatenate([get_std_across_users(hurdle_bern_param_array), \
    np.mean(get_std_across_users(zip_bern_param_array))], axis=None)
    zip_y_std = np.concatenate([get_std_across_users(hurdle_bern_param_array), \
    np.mean(get_std_across_users(zip_y_param_array))], axis=None)

    print("HURDLE MEAN", hurdle_bern_mean_vector)
    print("HURDLE MEAN", hurdle_y_mean_vector)
    print("HURDLE STD", hurdle_bern_std_diag)
    print("HURDLE STD", hurdle_y_std_diag)
    print("ZIP MEAN", zip_bern_mean)
    print("ZIP STD", zip_bern_std)
    print("ZIP MEAN", zip_y_mean)
    print("ZIP STD", zip_y_std)

    ## simulating the effect sizes per user ##
    user_effect_sizes = {}
    np.random.seed(1)
    for user in parameter_df['User']:
        if np.array(parameter_df[parameter_df['User'] == user])[0][2] == "sqrt_norm":
            bern_eff_size = np.random.multivariate_normal(hurdle_bern_mean_vector, np.diag(hurdle_bern_std_diag))
            y_eff_size = np.random.multivariate_normal(hurdle_y_mean_vector, np.diag(hurdle_y_std_diag))
        else:
            bern_eff_size = np.random.multivariate_normal(zip_bern_mean, np.diag(zip_bern_std))
            y_eff_size = np.random.multivariate_normal(zip_y_mean, np.diag(zip_y_std))

        user_effect_sizes[user] = [bern_eff_size, y_eff_size]

    return user_effect_sizes

STAT_USER_EFFECT_SIZES = get_effect_sizes(ROBAS_3_STAT_PARAMS_DF, BERN_PARAM_TITLES[:2] + BERN_PARAM_TITLES[3:], \
Y_PARAM_TITLES[:2] + Y_PARAM_TITLES[3:])
NON_STAT_USER_EFFECT_SIZES = get_effect_sizes(ROBAS_3_NON_STAT_PARAMS_DF, BERN_PARAM_TITLES, Y_PARAM_TITLES)

print("STAT USER EFFECT SIZES: ", STAT_USER_EFFECT_SIZES)
print("NON STAT USER EFFECT SIZES: ", NON_STAT_USER_EFFECT_SIZES)

with open("sim_env_data/stat_user_effect_sizes.p", 'wb') as f:
    pickle.dump(STAT_USER_EFFECT_SIZES, f)
with open("sim_env_data/non_stat_user_effect_sizes.p", 'wb') as f:
    pickle.dump(NON_STAT_USER_EFFECT_SIZES, f)
