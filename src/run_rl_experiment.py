from absl import app
from absl import flags
import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import pickle
import numpy as np
import pandas as pd
import copy

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('seed', None, 'input the seed value')
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('alg_type', None, 'input the RL algorithm candidate type')
flags.DEFINE_string('clipping_vals', None, 'input the clipping values')
flags.DEFINE_string('b_logistic', None, 'input the slope for the smoothing function')
flags.DEFINE_string('update_cadence', None, 'input the number of decision times before the next update')
flags.DEFINE_string('cluster_size', None, 'input the cluster_size')
flags.DEFINE_string('effect_size_scale', None, 'input the effect size scale')
# hyperparameter tuning
flags.DEFINE_string('tuning_hypers', None, 'input whether or not we are doing hyperparameter tuning')
flags.DEFINE_string('algorithm_val_1', None, 'input the RL algorithm candidate value for var 1')
flags.DEFINE_string('algorithm_val_2', None, 'input the RL algorithm candidate value for var 2')

MAX_SEED_VAL = 100
NUM_TRIAL_USERS = 72

def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

def get_sim_env(current_seed):
    # draw different users per trial
    print("SEED: ", current_seed)
    np.random.seed(current_seed)
    study_idxs = np.random.choice(simulation_environment.NUM_USERS, size=NUM_TRIAL_USERS)

    # get user ids corresponding to index
    users_list = get_user_list(study_idxs)
    print(users_list)

    ## HANDLING SIMULATION ENVIRONMENT ##
    env_type = FLAGS.sim_env_type
    # type String ["smaller", "small"]
    effect_size_scale = FLAGS.effect_size_scale
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

    print("PROCESSED ENV_TYPE {}".format(env_type))

    return users_list, environment_module

# parses argv to access FLAGS
def main(_argv):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    L_min = float(FLAGS.clipping_vals.split("_")[0])
    L_max = float(FLAGS.clipping_vals.split("_")[1])
    b_logistic = float(FLAGS.b_logistic)
    print("CLIPPING VALUES: [{}, {}]".format(L_min, L_max))
    smoothing_func_candidate = smoothing_function.genearlized_logistic_func_wrapper(L_min, L_max, b_logistic)
    update_cadence = int(FLAGS.update_cadence)
    cost_params = [int(FLAGS.algorithm_val_1), int(FLAGS.algorithm_val_2)]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    if FLAGS.alg_type == 'BLR_AC':
        alg_candidate = rl_algorithm.BlrActionCentering(cost_params, update_cadence, smoothing_func_candidate)
    elif FLAGS.alg_type == 'BLR_NO_AC':
        alg_candidate = rl_algorithm.BlrNoActionCentering(cost_params, update_cadence, smoothing_func_candidate)
    else:
        print("ERROR: NO ALG_TYPE FOUND - ", FLAGS.alg_type)
    print("ALG TYPE: {}".format(FLAGS.alg_type))

    # check if we are hyperparameter tuning or evaluating algorithm candidates
    if bool(FLAGS.tuning_hypers):
        print("We are doing hyperparameter tuning!")
        pickle_names = (FLAGS.sim_env_type, FLAGS.effect_size_scale, cost_params[0], cost_params[1])
        data_pickle_template = 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_data_df.p'
        update_pickle_template = 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_update_df.p'
        estimating_eqns_pickle_template = 'hyper_pickle_results/{}_{}_{}_{}_'.format(*pickle_names) + '{}_estimating_eqns_df.p'
    else:
        pickle_names = (FLAGS.sim_env_type, FLAGS.effect_size_scale, FLAGS.alg_type, FLAGS.b_logistic, FLAGS.clipping_vals, FLAGS.update_cadence, FLAGS.cluster_size)
        data_pickle_template = 'pickle_results/{}_{}_{}_{}_{}_{}_{}_'.format(*pickle_names) + '{}_data_df.p'
        update_pickle_template = 'pickle_results/{}_{}_{}_{}_{}_{}_{}_'.format(*pickle_names) + '{}_update_df.p'
        estimating_eqns_pickle_template = 'pickle_results/{}_{}_{}_{}_{}_{}_{}_'.format(*pickle_names) + '{}_estimating_eqns_df.p'

    cluster_size = int(FLAGS.cluster_size)
    if cluster_size == 1:
        alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(NUM_TRIAL_USERS)]
        for current_seed in range(MAX_SEED_VAL):

            _, environment_module = get_sim_env(current_seed)
            data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)

    elif cluster_size == NUM_TRIAL_USERS:
        for current_seed in range(MAX_SEED_VAL):
            users_list, environment_module = get_sim_env(current_seed)
            user_groups = rl_experiments.pre_process_users(users_list)
            data_df, update_df, estimating_eqns_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)
            data_df_pickle_location = data_pickle_template.format(current_seed)
            update_df_pickle_location = update_pickle_template.format(current_seed)
            estimating_eqns_df_pickle_location = estimating_eqns_pickle_template.format(current_seed)

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, data_df_pickle_location)
            pd.to_pickle(update_df, update_df_pickle_location)
            pd.to_pickle(estimating_eqns_df, estimating_eqns_df_pickle_location)

if __name__ == '__main__':
    app.run(main)
