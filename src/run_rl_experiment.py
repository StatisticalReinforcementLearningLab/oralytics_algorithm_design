from absl import app
from absl import flags
import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import pickle
import numpy as np
import pandas as pd

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('seed', None, 'input the seed value')
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('clipping_vals', None, 'input the clipping values')
flags.DEFINE_string('b_logistic', None, 'input the slope for the smoothing function')
flags.DEFINE_string('alg_type', None, 'input the RL algorithm candidate type')
flags.DEFINE_string('update_cadence', None, 'input the number of decision times before the next update')

MAX_SEED_VAL = 100
NUM_TRIAL_USERS = 72

def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

def run_experiment(alg_candidate, current_seed):
    # draw different users per trial
    print("SEED: ", current_seed)
    STUDY_IDXS = np.random.choice(simulation_environment.NUM_USERS, size=NUM_TRIAL_USERS)

    # get user ids corresponding to index
    USERS_LIST = get_user_list(STUDY_IDXS)
    print(USERS_LIST)

    ## HANDLING SIMULATION ENVIRONMENT ##
    env_type = FLAGS.sim_env_type
    if env_type == 'STAT_LOW_R':
        environment_module = simulation_environment.STAT_LOW_R(USERS_LIST)
    elif env_type == 'STAT_MED_R':
        environment_module = simulation_environment.STAT_MED_R(USERS_LIST)
    elif env_type == 'STAT_HIGH_R':
        environment_module = simulation_environment.STAT_HIGH_R(USERS_LIST)
    elif env_type == 'NON_STAT_LOW_R':
        environment_module = simulation_environment.NON_STAT_LOW_R(USERS_LIST)
    elif env_type == 'NON_STAT_MED_R':
        environment_module = simulation_environment.NON_STAT_MED_R(USERS_LIST)
    elif env_type == 'NON_STAT_HIGH_R':
        environment_module = simulation_environment.NON_STAT_HIGH_R(USERS_LIST)
    else:
        print("ERROR: NO ENV_TYPE FOUND - ", env_type)

    print("PROCESSED ENV_TYPE {}".format(env_type))

    ## RUN EXPERIMENT ##
    # Full Pooling with Incremental Recruitment
    user_groups = rl_experiments.pre_process_users(USERS_LIST)
    data_df, update_df, estimating_eqns_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)

    return data_df, update_df, estimating_eqns_df

# parses argv to access FLAGS
def main(_argv):
    ## HANDLING RL ALGORITHM CANDIDATE ##
    L_min = float(FLAGS.clipping_vals.split("_")[0])
    L_max = float(FLAGS.clipping_vals.split("_")[1])
    b_logistic = float(FLAGS.b_logistic)
    print("CLIPPING VALUES: [{}, {}]".format(L_min, L_max))
    smoothing_func_candidate = smoothing_function.genearlized_logistic_func_wrapper(L_min, L_max, b_logistic)
    update_cadence = int(FLAGS.update_cadence)
    if FLAGS.alg_type == 'BLR_AC':
        alg_candidate = rl_algorithm.BlrActionCentering([100, 100], update_cadence, smoothing_func_candidate)
    elif FLAGS.alg_type == 'BLR_NO_AC':
        alg_candidate = rl_algorithm.BlrNoActionCentering([100, 100], update_cadence, smoothing_func_candidate)
    else:
        print("ERROR: NO ALG_TYPE FOUND - ", FLAGS.alg_type)
    print("ALG TYPE: {}".format(FLAGS.alg_type))

    for current_seed in range(MAX_SEED_VAL):
        np.random.seed(current_seed)
        data_df, update_df, estimating_eqns_df = run_experiment(alg_candidate, current_seed)
        data_df_pickle_location = 'pickle_results/{}_{}_{}_{}_data_df.p'.format(FLAGS.sim_env_type, FLAGS.alg_type, FLAGS.b_logistic, current_seed)
        update_df_pickle_location = 'pickle_results/{}_{}_{}_{}_update_df.p'.format(FLAGS.sim_env_type, FLAGS.alg_type, FLAGS.b_logistic, current_seed)
        estimating_eqns_df_pickle_location = 'pickle_results/{}_{}_{}_{}_estimating_eqns_df.p'.format(FLAGS.sim_env_type, FLAGS.alg_type, FLAGS.b_logistic, current_seed)

        print("TRIAL DONE, PICKLING NOW")
        pd.to_pickle(data_df, data_df_pickle_location)
        pd.to_pickle(update_df, update_df_pickle_location)
        pd.to_pickle(estimating_eqns_df, estimating_eqns_df_pickle_location)

if __name__ == '__main__':
    app.run(main)
