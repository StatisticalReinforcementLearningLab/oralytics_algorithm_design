import rl_experiments
import rl_algorithm
import sim_env_v1
import sim_env_v2
import sim_env_v3
import smoothing_function
import experiment_global_vars
import compute_metrics

import pickle
import numpy as np
import pandas as pd
import copy

### experiment parameters ###
'''
### Simulation Env. Variants
* 'base_env_type': 'simulation environment type, stationary (STAT) or non-stationary (NON_STAT)'
* 'effect_size_scale': 'scales the imputed base treatment effect size'
* 'delayed_effect_scale': 'shrinks the treatment effect size to simulate delayed effects of actions'
### Algorithm Candidates
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

MAX_SEED_VAL = experiment_global_vars.MAX_SEED_VAL
NUM_TRIAL_USERS = experiment_global_vars.NUM_TRIAL_USERS

def get_user_list(version, study_idxs):
    user_list = [version.USER_INDICES[idx] for idx in study_idxs]

    return user_list

def get_sim_env(sim_env_version, base_env_type, effect_size_scale, delayed_effect_scale, current_seed):
    if sim_env_version == "v3":
        version = sim_env_v3
        sim_env = sim_env_v3.SimulationEnvironmentV3
    elif sim_env_version == "v2":
        version = sim_env_v2
        sim_env = sim_env_v2.SimulationEnvironmentV2
    elif sim_env_version == "v1":
        version = sim_env_v1
        sim_env = sim_env_v1.SimulationEnvironmentV1
    else:
        print("ERROR: NO SIM ENV VERSION FOUND - ", sim_env_version)
    # draw different users per trial
    print("SEED: ", current_seed)
    np.random.seed(current_seed)
    study_idxs = np.random.choice(version.NUM_USER_MODELS, size=NUM_TRIAL_USERS)

    # get user ids corresponding to index
    users_list = get_user_list(version, study_idxs)
    print(users_list)
    ## HANDLING SIMULATION ENVIRONMENT ##
    environment_module = sim_env(users_list, base_env_type, effect_size_scale, delayed_effect_scale)

    print("PROCESSED ENV_TYPE: {}, EFFECT SIZE SCALE: {}".format(base_env_type, effect_size_scale))

    return users_list, environment_module

def get_alg_candidate(alg_type, cluster_size, smoothing_func, update_cadence, cost_params, noise_var):
    if alg_type == 'BLR_AC':
        alg_candidate = rl_algorithm.BlrActionCentering(cost_params, update_cadence, smoothing_func, noise_var)
    elif alg_type == 'BLR_NO_AC':
        alg_candidate = rl_algorithm.BlrNoActionCentering(cost_params, update_cadence, smoothing_func, noise_var)
    elif alg_type == 'BLR_AC_V2':
        alg_candidate = rl_algorithm.BlrACV2(cost_params, update_cadence, smoothing_func)
    elif alg_type == 'BLR_AC_V3':
        alg_candidate = rl_algorithm.BlrACV3(cost_params, update_cadence, smoothing_func)
    else:
        print("ERROR: NO ALG_TYPE FOUND - ", alg_type)
    print("ALG TYPE: {}".format(alg_type))
    return alg_candidate

def get_cluster_size(pooling_type):
    if pooling_type == "full_pooling":
        return NUM_TRIAL_USERS
    elif pooling_type == "no_pooling":
        return 1
    else:
        print("ERROR: NO CLUSTER_SIZE FOUND - ", pooling_type)

# ANNA TODO: need to do similar procedure for reward def. tuning
# check if we are hyperparameter tuning or evaluating algorithm candidates
# if tuning_hypers:
    # print("We are doing hyperparameter tuning!")
    # pickle_names = (exp_kwargs["sim_env_type"], exp_kwargs["effect_size_scale"], cost_params[0], cost_params[1])
    # data_pickle_template = exp_path + 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_data_df.p'
    # update_pickle_template = exp_path + 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_update_df.p'
#"compute_metrics" #hyper_tuning
def run_experiment(exp_kwargs, exp_path, job_type):
    if job_type == "simulations":
        run_simulations(exp_kwargs, exp_path)
    elif job_type == "compute_metrics":
        run_compute_metrics(exp_kwargs, exp_path)
    elif job_type == "hyper_tuning":
        run_hyperparameter_tuning(exp_kwargs, exp_path)
    else:
        print("ERROR: NO JOB_TYPE FOUND - ", job_type)

def run_hyperparameter_tuning(exp_kwargs, exp_path):
    cluster_size = get_cluster_size("full_pooling")
    L_min, L_max = [0.2, 0.8]
    b_logistic = 0.515
    smoothing_func = smoothing_function.genearlized_logistic_func_wrapper(L_min, L_max, b_logistic)
    update_cadence = 14
    cost_params = exp_kwargs["cost_params"]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    alg_type = exp_kwargs["alg_type"]
    alg_candidate = get_alg_candidate(alg_type, cluster_size, smoothing_func, update_cadence, cost_params, None)

    data_pickle_template = exp_path + '/{}_data_df.p'
    update_pickle_template = exp_path + '/{}_update_df.p'

    sim_env_version = exp_kwargs["sim_env_version"]
    base_env_type = exp_kwargs["base_env_type"]
    effect_size_scale = exp_kwargs["effect_size_scale"]
    delayed_effect_scale = exp_kwargs["delayed_effect_scale"]

    for current_seed in range(MAX_SEED_VAL):
        users_list, environment_module = get_sim_env(sim_env_version, base_env_type, effect_size_scale, delayed_effect_scale, current_seed)
        user_groups = rl_experiments.pre_process_users(users_list)
        data_df, update_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)
        data_df_pickle_location = data_pickle_template.format(current_seed)
        update_df_pickle_location = update_pickle_template.format(current_seed)

        print("TRIAL DONE, PICKLING NOW")
        pd.to_pickle(data_df, data_df_pickle_location)
        pd.to_pickle(update_df, update_df_pickle_location)

def run_simulations(exp_kwargs, exp_path):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = get_cluster_size(exp_kwargs["cluster_size"])
    L_min, L_max = exp_kwargs["clipping_vals"]
    b_logistic = exp_kwargs["b_logistic"]
    print("CLIPPING VALUES: [{}, {}]".format(L_min, L_max))
    smoothing_func = smoothing_function.genearlized_logistic_func_wrapper(L_min, L_max, b_logistic)
    update_cadence = exp_kwargs["update_cadence"]
    cost_params = exp_kwargs["cost_params"]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    noise_var = exp_kwargs["noise_var"]
    alg_type = exp_kwargs["alg_type"]
    alg_candidate = get_alg_candidate(alg_type, cluster_size, smoothing_func, update_cadence, cost_params, noise_var)

    data_pickle_template = exp_path + '/{}_data_df.p'
    update_pickle_template = exp_path + '/{}_update_df.p'

    sim_env_version = exp_kwargs["sim_env_version"]
    base_env_type = exp_kwargs["base_env_type"]
    effect_size_scale = exp_kwargs["effect_size_scale"]
    delayed_effect_scale = exp_kwargs["delayed_effect_scale"]

    if cluster_size == 1:
        alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(NUM_TRIAL_USERS)]
        for current_seed in range(MAX_SEED_VAL):

            _, environment_module = get_sim_env(sim_env_version, base_env_type, effect_size_scale, delayed_effect_scale, current_seed)
            data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)

    elif cluster_size == NUM_TRIAL_USERS:
        for current_seed in range(MAX_SEED_VAL):
            users_list, environment_module = get_sim_env(sim_env_version, base_env_type, effect_size_scale, delayed_effect_scale, current_seed)
            user_groups = rl_experiments.pre_process_users(users_list)
            data_df, update_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)

def run_compute_metrics(exp_kwargs, exp_path):
    env_alg_mean, env_alg_lower_25 = compute_metrics.get_metric_values(exp_path, MAX_SEED_VAL)

    avg_pickle_location = exp_path + '/avg.p'
    lower_25_pickle_location = exp_path + '/low_25.p'

    with open(avg_pickle_location, 'wb') as handle:
        pickle.dump(env_alg_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(lower_25_pickle_location, 'wb') as handle:
        pickle.dump(env_alg_lower_25, handle, protocol=pickle.HIGHEST_PROTOCOL)
