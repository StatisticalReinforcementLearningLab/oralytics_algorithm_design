import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import pickle
import pandas as pd
import numpy as np
import copy

# NUM_TRIAL_USERS must be a multiple of 4 which is the recruitment rate
NUM_TRIAL_USERS = 8
def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

MAX_SEED_VAL = 1

def main():
    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = NUM_TRIAL_USERS
    # cluster_size = 1
    smoothing_func_candidate = smoothing_function.genearlized_logistic_func_wrapper(0.1, 0.9, 0.515)
    alg_candidate = rl_algorithm.BlrActionCentering([100, 100], 2, smoothing_func_candidate)

    for current_seed in range(MAX_SEED_VAL):
        # draw different users per trial
        np.random.seed(current_seed)
        print("SEED: ", current_seed)

        STUDY_IDXS = np.random.choice(simulation_environment.NUM_USERS, size=NUM_TRIAL_USERS)
        print(STUDY_IDXS)

        # get user ids corresponding to index
        USERS_LIST = get_user_list(STUDY_IDXS)
        print(USERS_LIST)

        ## HANDLING SIMULATION ENVIRONMENT ##
        environment_module = simulation_environment.STAT_LOW_R(USERS_LIST)
        # environment_module = STAT_MED_R(USERS_LIST)
        # environment_module = STAT_HIGH_R(USERS_LIST)
        # environment_module = NON_STAT_LOW_R(USERS_LIST)
        # environment_module = NON_STAT_MED_R(USERS_LIST)
        # environment_module = NON_STAT_HIGH_R(USERS_LIST)

        ## RUN EXPERIMENT ##
        # No Pooling
        if cluster_size == 1:
            alg_candidates = [copy.deepcopy(alg_candidate) for _ in range(NUM_TRIAL_USERS)]
            data_df, update_df = rl_experiments.run_experiment(alg_candidates, environment_module)

            data_df.to_csv("pickle_results/test_data_df.csv")
            update_df.to_csv("pickle_results/test_update_df.csv")

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, 'pickle_results/{}_{}_data_df.p'.format("test", current_seed))
            pd.to_pickle(update_df, 'pickle_results/{}_{}_update_df.p'.format("test", current_seed))

        # Full Pooling with Incremental Recruitment
        elif cluster_size == NUM_TRIAL_USERS:
            user_groups = rl_experiments.pre_process_users(USERS_LIST)
            data_df, update_df, estimating_eqns_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)

            data_df.to_csv("pickle_results/test_data_df.csv")
            update_df.to_csv("pickle_results/test_update_df.csv")
            estimating_eqns_df.to_csv("pickle_results/test_estimating_eqns_df.csv")

            print("TRIAL DONE, PICKLING NOW")
            pd.to_pickle(data_df, 'pickle_results/{}_{}_data_df.p'.format("test", current_seed))
            pd.to_pickle(update_df, 'pickle_results/{}_{}_update_df.p'.format("test", current_seed))
            pd.to_pickle(update_df, 'pickle_results/{}_{}_estimating_eqns_df.p'.format("test", current_seed))


main()
