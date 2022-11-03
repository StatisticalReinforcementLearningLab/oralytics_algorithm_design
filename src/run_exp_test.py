import rl_experiments
import rl_algorithm
import simulation_environment
import smoothing_function
import pickle
import numpy as np

NUM_TRIAL_USERS = 72
def get_user_list(study_idxs):
    user_list = [simulation_environment.USER_INDICES[idx] for idx in study_idxs]

    return user_list

MAX_SEED_VAL = 1

def main():
    ## HANDLING RL ALGORITHM CANDIDATE ##
    alg_candidate = rl_algorithm.BlrActionCentering([100, 100], 13, smoothing_function.BASIC_THOMPSON_SAMPLING_FUNC)
    # alg_candidate = rl_algorithm.BlrNoActionCentering([100, 100], 13, rl_algorithm.GENERALIZED_LOGISTIC_FUNC)

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
        # Full Pooling with Incremental Recruitment
        user_groups = rl_experiments.pre_process_users(USERS_LIST)
        data_df, update_df, estimating_eqns_df = rl_experiments.run_incremental_recruitment_exp(user_groups, alg_candidate, environment_module)

        data_df.to_csv("pickle_results/test_data_df.csv")
        update_df.to_csv("pickle_results/test_update_df.csv")
        estimating_eqns_df.to_csv("pickle_results/test_estimating_eqns_df.csv")

        data_df_pickle_location = 'pickle_results/{}_{}_data_df.p'.format("test", current_seed)
        update_df_pickle_location = 'pickle_results/{}_{}_update_df.p'.format("test", current_seed)
        estimating_eqns_df_pickle_location = 'pickle_results/{}_{}_estimating_eqns_df.p'.format("test", current_seed)

        print("TRIAL DONE")
        with open(data_df_pickle_location, 'wb') as f:
            pickle.dump(data_df, f)
        with open(update_df_pickle_location, 'wb') as f:
            pickle.dump(update_df, f)
        with open(estimating_eqns_df_pickle_location, 'wb') as f:
            pickle.dump(estimating_eqns_df, f)

main()
