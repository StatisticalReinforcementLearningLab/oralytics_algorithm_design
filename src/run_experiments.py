import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import read_write_info
import pickle
import numpy as np
import pandas as pd
import copy

### experiment parameters ###
'''
* 'sim_env_type': 'simulation environment type'
* 'effect_size_scale': 'scales the imputed treatment effect size'
* 'alg_type': 'RL algorithm candidate type'
* 'noise_var': noise variance value
* 'clipping_vals': 'asymptotes for the smoothing function'
* 'b_logistic': 'the slope for the smoothing function'
* 'update_cadence': 'number of decision times before the next update'
* 'cluster_size': 'number of users within a cluster'
'''
### hyperparameter tuning for reward definition ###
'''
* 'tuning_hypers': whether or not we are doing hyperparameter tuning
* 'algorithm_val_1': the RL algorithm candidate value for var 1
* 'algorithm_val_2': the RL algorithm candidate value for var 2
'''

MAX_SEED_VAL = 100
NUM_TRIAL_USERS = 72

def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

def get_sim_env(env_type, effect_size_scale, current_seed):
    # draw different users per trial
    print("SEED: ", current_seed)
    np.random.seed(current_seed)
    study_idxs = np.random.choice(simulation_environment.NUM_USERS, size=NUM_TRIAL_USERS)

    # get user ids corresponding to index
    users_list = get_user_list(study_idxs)
    print(users_list)

    ## HANDLING SIMULATION ENVIRONMENT ##
    if env_type == 'STAT_LOW_R':
        environment_module = simulation_environment.STAT_LOW_R(users_list, effect_size_scale)
    elif env_type == 'STAT_MED_R':
        environment_module = simulation_environment.STAT_MED_R(users_list, effect_size_scale)
    elif env_type == 'STAT_HIGH_R':
        environment_module = simulation_environment.STAT_HIGH_R(users_list, effect_size_scale)
    elif env_type == 'NON_STAT_LOW_R':
        environment_module = simulation_environment.NON_STAT_LOW_R(users_list, effect_size_scale)
    elif env_type == 'NON_STAT_MED_R':
        environment_module = simulation_environment.NON_STAT_MED_R(users_list, effect_size_scale)
    elif env_type == 'NON_STAT_HIGH_R':
        environment_module = simulation_environment.NON_STAT_HIGH_R(users_list, effect_size_scale)
    else:
        print("ERROR: NO ENV_TYPE FOUND - ", env_type)

    print("PROCESSED ENV_TYPE: {}, EFFECT SIZE SCALE: {}".format(env_type, effect_size_scale))

    return users_list, environment_module

# ANNA TODO: need to do similar procedure for reward def. tuning
# check if we are hyperparameter tuning or evaluating algorithm candidates
# if tuning_hypers:
    # print("We are doing hyperparameter tuning!")
    # pickle_names = (exp_kwargs["sim_env_type"], exp_kwargs["effect_size_scale"], cost_params[0], cost_params[1])
    # data_pickle_template = exp_path + 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_data_df.p'
    # update_pickle_template = exp_path + 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_update_df.p'

def run_experiment(exp_kwargs, exp_path):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = exp_kwargs["cluster_size"]
    L_min, L_max = exp_kwargs["clipping_vals"]
    b_logistic = exp_kwargs["b_logistic"]
    print("CLIPPING VALUES: [{}, {}]".format(L_min, L_max))
    smoothing_func_candidate = smoothing_function.genearlized_logistic_func_wrapper(L_min, L_max, b_logistic)
    update_cadence = exp_kwargs["update_cadence"]
    cost_params = exp_kwargs["cost_params"]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    noise_var = exp_kwargs["noise_var"]
    if exp_kwargs["alg_type"] == 'BLR_AC':
        alg_candidate = rl_algorithm.BlrActionCentering(cost_params, update_cadence, smoothing_func_candidate, noise_var)
    elif exp_kwargs["alg_type"] == 'BLR_NO_AC':
        alg_candidate = rl_algorithm.BlrNoActionCentering(cost_params, update_cadence, smoothing_func_candidate, noise_var)
    else:
        print("ERROR: NO ALG_TYPE FOUND - ", exp_kwargs["alg_type"])
    print("ALG TYPE: {}".format(exp_kwargs["alg_type"]))

    data_pickle_template = exp_path + '/{}_data_df.p'
    update_pickle_template = exp_path + '/{}_update_df.p'

    env_type = exp_kwargs["sim_env_type"]
    effect_size_scale = exp_kwargs["effect_size_scale"]

    if cluster_size == 1:
        alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(NUM_TRIAL_USERS)]
        for current_seed in range(MAX_SEED_VAL):

            _, environment_module = get_sim_env(env_type, effect_size_scale, current_seed)
            data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)

    elif cluster_size == NUM_TRIAL_USERS:
        for current_seed in range(MAX_SEED_VAL):
            users_list, environment_module = get_sim_env(env_type, effect_size_scale, current_seed)
            user_groups = rl_experiments.pre_process_users(users_list)
            data_df, update_df, _ = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)

if __name__ == '__main__':
    app.run(main)
