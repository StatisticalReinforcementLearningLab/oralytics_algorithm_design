from absl import app
from absl import flags
import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import pickle
import numpy as np

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('seed', None, 'input the seed value')
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('clipping_vals', None, 'input the clipping values')
flags.DEFINE_string('b_logistic', None, 'input the slope for the smoothing function')
# flags.DEFINE_string('rl_algorithm_type', None, 'input the RL algorithm candidate type')
MAX_SEED_VAL = 100
NUM_TRIAL_USERS = 72

def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

def run_experiment(alg_candidate, current_seed):
    # denotes a weekly update schedule
    UPDATE_CADENCE = 13

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
    alg_candidate = rl_algorithm.BlrActionCentering([100, 100], 13, smoothing_func_candidate)

    for current_seed in range(MAX_SEED_VAL):
        np.random.seed(current_seed)
        data_df, update_df, estimating_eqns_df = run_experiment(alg_candidate, current_seed)
        data_df_pickle_location = 'pickle_results/{}_{}_{}_{}_data_df.p'.format(FLAGS.sim_env_type, FLAGS.clipping_vals, FLAGS.b_logistic, current_seed)
        update_df_pickle_location = 'pickle_results/{}_{}_{}_{}_update_df.p'.format(FLAGS.sim_env_type, FLAGS.clipping_vals, FLAGS.b_logistic, current_seed)
        estimating_eqns_df_pickle_location = 'pickle_results/{}_{}_{}_{}_estimating_eqns_df.p'.format(FLAGS.sim_env_type, FLAGS.clipping_vals, FLAGS.b_logistic, current_seed)

        ## results is a list of tuples where the first element of the tuple is user_id and the second element is a dictionary of values
        print("TRIAL DONE, PICKLING NOW")
        with open(data_df_pickle_location, 'wb') as f:
            pickle.dump(data_df, f)
        with open(update_df_pickle_location, 'wb') as f:
            pickle.dump(update_df, f)
        with open(estimating_eqns_df_pickle_location, 'wb') as f:
            pickle.dump(estimating_eqns_df, f)

if __name__ == '__main__':
    app.run(main)
