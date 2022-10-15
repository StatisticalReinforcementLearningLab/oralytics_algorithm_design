from code import simulation_environment
from code import rl_algorithm
import numpy as np
import pandas as pd

## GLOBAL VALUES ##
### ANNA TODO: FOR INITIAL EXPERIMENTS WE ARE NOT DOING INCREMENTAL RECRUITEMENT ###
### CHANGE BACK TO  `RECRUITMENT_RATE = 4` TO DO INCREMENTAL RECRUITMENT ###
RECRUITMENT_RATE = 72
TRIAL_LENGTH_IN_WEEKS = 10
# We should have NUM_USERS x NUM_DECISION_TIMES datapoints for each saved value or
# statistic at the end of the study
NUM_DECISION_TIMES = 70 * 2
# BATCH_DATA_SIZE = 72 * NUM_DECISION_TIMES

# assumes a weekly recruitment rate
def compute_num_updates(users_groups, update_cadence):
    last_group_idx = max(users_groups[:,1].astype(int))
    num_study_decision_times = (last_group_idx + TRIAL_LENGTH_IN_WEEKS) * 7 * 2
    # we subtract 1 because we do not update after the final week of the study
    num_updates = (num_study_decision_times / update_cadence) - 1

    return int(num_updates)

FILL_IN_COLS = ['action', 'prob', 'reward', 'quality', 'state.tod', 'state.b.bar',\
 'state.a.bar', 'state.day.type', 'state.bias']

def create_dfs(users_groups, update_cadence, rl_algorithm_feature_dim):
    N = len(users_groups)
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_idx'] = np.repeat(users_groups[:,0], NUM_DECISION_TIMES)
    data_dict['user_id'] = np.repeat(users_groups[:,2], NUM_DECISION_TIMES)
    data_dict['decision time'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['day in study'] = 1 + ((14 * np.repeat(users_groups[:,1].astype(int), NUM_DECISION_TIMES) + data_dict['decision time']) // 2)
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    ### udpate df ###
    update_dict = {}
    num_updates = compute_num_updates(users_groups, update_cadence)
    update_dict['update_t'] = np.arange(1, num_updates + 1)
    for i in range(rl_algorithm_feature_dim):
        update_dict['posterior_mu.{}'.format(i)] = np.full(num_updates, np.nan)
    for i in range(rl_algorithm_feature_dim):
        for j in range(rl_algorithm_feature_dim):
            update_dict['posterior_var.{}.{}'.format(i, j)] = np.full(num_updates, np.nan)
    update_df = pd.DataFrame.from_dict(update_dict)
    ### estimating eqns df ###
    estimating_eqns_dict = {}
    num_updates_for_cluster = int(NUM_DECISION_TIMES / update_cadence)
    last_group_idx = max(users_groups[:,1].astype(int))
    estimating_eqns_dict['update_t'] = np.stack([range(1 + start_idx, start_idx + num_updates_for_cluster + 1) \
    for start_idx in users_groups[:,1].astype(int)], axis=0).flatten()
    estimating_eqns_dict['user_idx'] = np.repeat(users_groups[:,0], num_updates_for_cluster)
    estimating_eqns_dict['user_id'] = np.repeat(users_groups[:,2], num_updates_for_cluster)
    for i in range(rl_algorithm_feature_dim):
        estimating_eqns_dict['mean_estimate.{}'.format(i)] = np.full(N * num_updates_for_cluster, np.nan)
        for j in range(rl_algorithm_feature_dim):
            estimating_eqns_dict['var_estimate.{}.{}'.format(i, j)] = np.full(N * num_updates_for_cluster, np.nan)
    estimating_eqns_df = pd.DataFrame.from_dict(estimating_eqns_dict)

    return data_df, update_df, estimating_eqns_df

# regex pattern '.*' gets you everything
def get_data_df_values_for_users(data_df, user_idxs, day_in_study, regex_pattern):
    return data_df.loc[(data_df['user_idx'].isin(user_idxs)) & (data_df['day in study'] <= day_in_study)].filter(regex=(regex_pattern))

def set_data_df_values_for_user(data_df, user_idx, decision_time, action, prob, reward, quality, alg_state):
    data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['decision time'] == decision_time), FILL_IN_COLS] = np.concatenate([[action, prob, reward, quality], alg_state])

def set_update_df_values(update_df, update_t, posterior_mu, posterior_var):
    update_df.iloc[update_df['update_t'] == update_t, 1:] = np.concatenate([posterior_mu, posterior_var.flatten()])

def set_estimating_eqns_df_values(df, update_t, user_idx, estimating_eqns):
    df.iloc[(df['update_t'] == update_t) & (df['user_idx'] == user_idx), 3:] = estimating_eqns

# n is the number of users currently in the study
# computes and sets estimating eqn statistic for every user currently in the study
def compute_and_estimating_equation_statistic(data_df, estimating_eqns_df, \
                                            current_groups, rl_algorithm, \
                                            update_t, day_in_study):
    n = len(current_groups)
    for i, user_tuple in enumerate(current_groups):
        user_idx = user_tuple[0]
        alg_state = get_data_df_values_for_users(data_df, [user_idx], "state.*")
        probs = get_data_df_values_for_users(data_df, [user_idx], "prob")
        actions = get_data_df_values_for_users(data_df, [user_idx], "action")
        big_phi = rl_algorithm.feature_map(np.delete(alg_state, 3),\
                                            alg_state,\
                                            probs,\
                                            actions)
        big_r = np.array(get_data_df_values_for_users(data_df, user_idx, 'reward'))
        estimating_eqn = rl_algorithm.compute_estimating_equation([big_phi, big_r], n)
        set_estimating_eqns_df_values(estimating_eqns_df, update_t, user_idx, estimating_eqn)

    return result


# returns a int(NUM_USERS / RECRUITMENT_RATE) x RECRUITMENT_RATE array of user indices
# row index represents the week that they enter the study
def pre_process_users(total_trial_users):
    results = []
    for j, user in enumerate(total_trial_users):
        results.append((j, int(j // RECRUITMENT_RATE), user))

    return np.array(results)

### runs experiment with full pooling and incremental recruitment
# users_groups will be a list of tuples where tuple[0] is the user index
# tuple[1] is the week they entered the study, tuple[2] is the user id string
def run_incremental_recruitment_exp(user_groups, alg_candidate, sim_env):
    env_users = sim_env.get_users()
    update_cadence = alg_candidate.get_update_cadence()
    data_df, update_df, estimating_eqns_df = create_dfs(users_groups, update_cadence, alg_candidate.feature_dim)
    current_groups = user_groups[:RECRUITMENT_RATE]
    week = 0
    while (len(current_groups) > 0):
        print("Week: ", week)
        for user_tuple in current_groups:
            user_idx, user_entry_date = int(user_tuple[0]), int(user_tuple[1])
            user_states = sim_env.get_states_for_user(user_idx)
            # do action selection for 14 decision times (7 days)
            ### ANNA TODO: need to generalize if we want to test different udpate cadences ###
            for decision_idx in range(14):
                ## PROCESS STATE ##
                j = (week - user_entry_date) * 14 + decision_idx
                env_state = sim_env.process_env_state(user_states[j], j, result[user_idx][1]["qualities"])
                # if first week for user, we impute A bar and B bar
                if j < 14:
                    b_bar = np.mean(obtain_subarray(result[user_idx][1]["qualities"])) \
                     if len(obtain_subarray(result[user_idx][1]["qualities"])) > 0 else 0
                    a_bar = np.mean(obtain_subarray(result[user_idx][1]["actions"])) \
                     if len(obtain_subarray(result[user_idx][1]["actions"])) > 0 else 0
                else:
                    b_bar = calculate_b_bar(result[user_idx][1]["qualities"][j - 14:j])
                    a_bar = calculate_a_bar(result[user_idx][1]["actions"][j - 14:j])
                advantage_state, baseline_state = alg_candidate.process_alg_state_func(env_state, b_bar, a_bar)
                ## SAVE STATE VALUES ##
                result[user_idx][1]["env_states"][j] = env_state.reshape(1, -1)
                result[user_idx][1]["advantage_states"][j] = advantage_state.reshape(1, -1)
                total_results["advantage_states"][current_batch_data_size] = advantage_state.reshape(1, -1)
                result[user_idx][1]["baseline_states"][j] = baseline_state.reshape(1, -1)
                total_results["baseline_states"][current_batch_data_size] = baseline_state.reshape(1, -1)
                ## ACTION SELECTION ##
                action, action_prob = alg_candidate.action_selection(advantage_state)
                ## SAVE ACTION VALUES ##
                result[user_idx][1]["actions"][j] = action
                total_results["actions"][current_batch_data_size] = action
                result[user_idx][1]["probs"][j] = action_prob
                total_results["probs"][current_batch_data_size] = action_prob
                ## REWARD GENERATION ##
                quality = sim_env.generate_rewards(user_idx, env_state, action)
                reward = alg_candidate.reward_def_func(quality, action, b_bar, a_bar)
                ## SAVE REWARD VALUES ##
                result[user_idx][1]["qualities"][j] = min(quality, 180)
                result[user_idx][1]["rewards"][j] = reward
                total_results["rewards"][current_batch_data_size] = reward
                # increment current_batch_data_size
                current_batch_data_size += 1
                ## UPDATE UNRESPONSIVENESS ##
                # if it's after the first week
                if j >= 14:
                    sim_env.update_responsiveness(user_idx, calculate_a1_condition(a_bar),\
                     calculate_a2_condition(a_bar), calculate_b_condition(b_bar), j)

        print("UPDATE TIME.")
        # calculate estimating equation statistic
        current_history = [result[int(i)] for i in current_groups[:,0]]
        statistic_results.append(compute_estimating_equation_statistic(current_history, len(current_groups), alg_candidate))
        # update time at the end of each week
        alg_candidate.update(total_results["advantage_states"][:current_batch_data_size], \
        total_results["baseline_states"][:current_batch_data_size], \
        total_results["actions"][:current_batch_data_size], \
        total_results["probs"][:current_batch_data_size], \
        total_results["rewards"][:current_batch_data_size])
        # handle adding or removing user groups
        week += 1
        if (week < len(user_groups) // RECRUITMENT_RATE):
            # add more users
            current_groups = np.concatenate((current_groups, user_groups[RECRUITMENT_RATE * week: RECRUITMENT_RATE * week + RECRUITMENT_RATE]), axis=0)
        # check if some user group finished the study
        if (week > TRIAL_LENGTH_IN_WEEKS - 1):
            current_groups = current_groups[RECRUITMENT_RATE:]

    return data_df, statistic_results
