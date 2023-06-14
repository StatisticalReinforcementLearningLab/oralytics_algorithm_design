import rl_algorithm
import numpy as np
import pandas as pd
import reward_definition
import experiment_global_vars

RECRUITMENT_RATE = experiment_global_vars.RECRUITMENT_RATE
TRIAL_LENGTH_IN_WEEKS = experiment_global_vars.TRIAL_LENGTH_IN_WEEKS
NUM_DECISION_TIMES = experiment_global_vars.NUM_DECISION_TIMES
FILL_IN_COLS = experiment_global_vars.FILL_IN_COLS

# assumes a weekly recruitment rate
def compute_num_updates(users_groups, update_cadence):
    last_group_idx = max(users_groups[:,1].astype(int))
    num_study_decision_times = (last_group_idx + TRIAL_LENGTH_IN_WEEKS) * 7 * 2
    # we subtract 1 because we do not update after the final week of the study
    num_updates = num_study_decision_times / update_cadence

    return int(num_updates)

def create_dfs_no_pooling(users, update_cadence, rl_algorithm_feature_dim):
    N = len(users)
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_idx'] = np.repeat(range(N), NUM_DECISION_TIMES)
    data_dict['user_id'] = np.repeat(users, NUM_DECISION_TIMES)
    data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['day_in_study'] = np.stack([1 + (np.arange(NUM_DECISION_TIMES) // 2) for _ in range(N)], axis=0).flatten()
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    ### udpate df ###
    update_dict = {}
    num_updates = int(NUM_DECISION_TIMES / update_cadence)
    update_dict['user_idx'] = np.repeat(range(N), num_updates)
    update_dict['user_id'] = np.repeat(users, num_updates)
    update_dict['update_t'] = np.stack([np.arange(0, num_updates) for _ in range(N)], axis=0).flatten()
    for i in range(rl_algorithm_feature_dim):
        update_dict['posterior_mu.{}'.format(i)] = np.full(N * num_updates, np.nan)
    for i in range(rl_algorithm_feature_dim):
        for j in range(rl_algorithm_feature_dim):
            update_dict['posterior_var.{}.{}'.format(i, j)] = np.full(N * num_updates, np.nan)
    update_df = pd.DataFrame.from_dict(update_dict)

    return data_df, update_df

def create_dfs_full_pooling(users_groups, update_cadence, rl_algorithm_feature_dim):
    N = len(users_groups)
    batch_data_size = N * NUM_DECISION_TIMES
    ### data df ###
    data_dict = {}
    data_dict['user_idx'] = np.repeat(users_groups[:,0].astype(int), NUM_DECISION_TIMES)
    data_dict['user_id'] = np.repeat(users_groups[:,2], NUM_DECISION_TIMES)
    data_dict['user_entry_decision_t'] = 14 * np.repeat(users_groups[:,1].astype(int), NUM_DECISION_TIMES)
    data_dict['user_last_decision_t'] = 14 * np.repeat(users_groups[:,1].astype(int), NUM_DECISION_TIMES) + (NUM_DECISION_TIMES - 1)
    data_dict['user_decision_t'] = np.stack([range(NUM_DECISION_TIMES) for _ in range(N)], axis=0).flatten()
    data_dict['calendar_decision_t'] = 14 * np.repeat(users_groups[:,1].astype(int), NUM_DECISION_TIMES) + data_dict['user_decision_t']
    data_dict['day_in_study'] = 1 + ((14 * np.repeat(users_groups[:,1].astype(int), NUM_DECISION_TIMES) + data_dict['user_decision_t']) // 2)
    for key in FILL_IN_COLS:
        data_dict[key] = np.full(batch_data_size, np.nan)
    data_df = pd.DataFrame.from_dict(data_dict)
    ### udpate df ###
    update_dict = {}
    num_updates = compute_num_updates(users_groups, update_cadence)
    update_dict['update_t'] = np.arange(0, num_updates)
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
    estimating_eqns_dict['update_t'] = np.stack([range(1 + start_idx * int(14 / update_cadence), start_idx * int(14 / update_cadence) + num_updates_for_cluster + 1) \
    for start_idx in users_groups[:,1].astype(int)], axis=0).flatten()
    estimating_eqns_dict['user_idx'] = np.repeat(users_groups[:,0].astype(int), num_updates_for_cluster)
    estimating_eqns_dict['user_id'] = np.repeat(users_groups[:,2], num_updates_for_cluster)
    for i in range(rl_algorithm_feature_dim):
        estimating_eqns_dict['mean_estimate.{}'.format(i)] = np.full(N * num_updates_for_cluster, np.nan)
    for i in range(rl_algorithm_feature_dim):
        for j in range(rl_algorithm_feature_dim):
            estimating_eqns_dict['var_estimate.{}.{}'.format(i, j)] = np.full(N * num_updates_for_cluster, np.nan)
    estimating_eqns_df = pd.DataFrame.from_dict(estimating_eqns_dict)

    return data_df, update_df, estimating_eqns_df

# regex pattern '.*' gets you everything
# Note: if regex pattern only refers to one column, then you need to .flatten() the resulting array
def get_data_df_values_for_users(data_df, user_idxs, day_in_study, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'].isin(user_idxs)) & (data_df['day_in_study'] <= day_in_study)].filter(regex=(regex_pattern)))

def get_user_data_values_from_decision_t(data_df, user_idx, decision_t, regex_pattern):
    return np.array(data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] < decision_t)].filter(regex=(regex_pattern)))

def set_data_df_values_for_user(data_df, user_idx, decision_time, policy_idx, action, prob, reward, quality, alg_state):
    data_df.loc[(data_df['user_idx'] == user_idx) & (data_df['user_decision_t'] == decision_time), FILL_IN_COLS] = np.concatenate([[policy_idx, action, prob, reward, quality], alg_state])

### for full pooling experiments ###
def set_update_df_values(update_df, update_t, posterior_mu, posterior_var):
    update_df.iloc[update_df['update_t'] == update_t, 1:] = np.concatenate([posterior_mu, posterior_var.flatten()])

### for no pooling experiments ###
def set_update_df_values_for_user(update_df, user_idx, update_t, posterior_mu, posterior_var):
    update_df.iloc[(update_df['update_t'] == update_t) & (update_df['user_idx'] == user_idx), 3:] = np.concatenate([posterior_mu, posterior_var.flatten()])

def set_estimating_eqns_df_values(df, update_t, user_idx, estimating_eqns):
    df.iloc[(df['update_t'] == update_t) & (df['user_idx'] == user_idx), 3:] = estimating_eqns

# n is the number of users currently in the study
# computes and sets estimating eqn statistic for every user currently in the study
def compute_and_estimating_equation_statistic(data_df, estimating_eqns_df, \
                                            current_groups, alg_candidate, \
                                            update_t, day_in_study):
    n = len(current_groups)
    for user_idx in current_groups:
        alg_state = get_data_df_values_for_users(data_df, [user_idx], day_in_study, "state.*")
        probs = get_data_df_values_for_users(data_df, [user_idx], day_in_study, "prob").flatten()
        actions = get_data_df_values_for_users(data_df, [user_idx], day_in_study, "action").flatten()
        big_phi = alg_candidate.feature_map(alg_state,\
                                            alg_state,\
                                            probs,\
                                            actions)
        big_r = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'reward').flatten()
        estimating_eqn = alg_candidate.compute_estimating_equation([big_phi, big_r], n)
        set_estimating_eqns_df_values(estimating_eqns_df, update_t, user_idx, estimating_eqn)

# if user did not open the app at all before the decision time, then we simulate
# the algorithm selecting action based off of a stale state (i.e., b_bar is the b_bar from when the user last opened their app)
# if user did open the app, then the algorithm selecting action based off of a fresh state (i.e., b_bar stays the same)
def get_alg_state_from_app_opening(user_last_open_app_dt, data_df, user_idx, j, advantage_state):

    # if morning dt we check if users opened the app in the morning
    # if evening dt we check if users opened the app in the morning and in the evening
    if j % 2 == 0:
        user_opened_app_today = (user_last_open_app_dt == j)
    else:
        # we only simulate users opening the app for morning dts
        user_opened_app_today = (user_last_open_app_dt == j - 1)
    if not user_opened_app_today:
        # impute b_bar with stale b_bar and prior day app engagement = 0
        stale_b_bar = get_user_data_values_from_decision_t(data_df, user_idx, user_last_open_app_dt + 1, 'state.b.bar').flatten()[-1]
        # refer to rl_algorithm.py process_alg_state functions for V2, V3
        advantage_state[1] = stale_b_bar
        advantage_state[3] = 0

    return advantage_state

def get_previous_day_qualities_and_actions(j, Qs, As):
    if j > 1:
        if j % 2 == 0:
            return Qs, As
        else:
            # current evening dt does not use most recent quality or action
            return Qs[:-1], As[:-1]
    # first day return empty Qs and As back
    else:
        return Qs, As

def execute_decision_time(data_df, user_idx, j, alg_candidate, sim_env, policy_idx):
    env_state = sim_env.generate_current_state(user_idx, j)
    user_qualities = get_user_data_values_from_decision_t(data_df, user_idx, j, 'quality').flatten()
    user_actions = get_user_data_values_from_decision_t(data_df, user_idx, j, 'action').flatten()
    Qs, As = get_previous_day_qualities_and_actions(j, user_qualities, user_actions)
    b_bar, a_bar = reward_definition.get_b_bar_a_bar(Qs, As)
    advantage_state, _ = alg_candidate.process_alg_state(env_state, b_bar, a_bar)
    # simulate app opening issue
    if sim_env.get_version() == "V2" or sim_env.get_version() == "V3":
        user_last_open_app_dt = sim_env.get_user_last_open_app_dt(user_idx)
        alg_state = get_alg_state_from_app_opening(user_last_open_app_dt, data_df, user_idx, j, advantage_state)
    else:
        alg_state = advantage_state
    ## ACTION SELECTION ##
    action, action_prob = alg_candidate.action_selection(alg_state)
    ## REWARD GENERATION ##
    # quality definition
    quality = sim_env.generate_rewards(user_idx, env_state, action)
    reward = alg_candidate.reward_def_func(quality, action, b_bar, a_bar)
    ## SAVE VALUES ##
    set_data_df_values_for_user(data_df, user_idx, j, policy_idx, action, action_prob, reward, quality, alg_state)
    ## UPDATE UNRESPONSIVENESS ##
    # if it's after the first week
    if j >= 14:
        sim_env.update_responsiveness(user_idx, reward_definition.calculate_a1_condition(a_bar),\
         reward_definition.calculate_a2_condition(a_bar), reward_definition.calculate_b_condition(b_bar), j)

def run_experiment(alg_candidates, sim_env):
    env_users = sim_env.get_users()
    # all alg_candidates have the same update cadence and feature dimension
    update_cadence = alg_candidates[0].get_update_cadence()
    data_df, update_df = create_dfs_no_pooling(env_users, update_cadence, alg_candidates[0].get_feature_dim())
    policy_idxs = np.zeros(len(env_users))
    # add in prior values to posterior dataframe
    for user_idx in range(len(env_users)):
        set_update_df_values_for_user(update_df, user_idx, 0, \
        alg_candidates[user_idx].posterior_mean, alg_candidates[user_idx].posterior_var)
    for j in range(NUM_DECISION_TIMES):
        # print("Decision Time: ", j)
        for user_idx in range(len(env_users)):
            alg_candidate = alg_candidates[user_idx]
            execute_decision_time(data_df, user_idx, j, alg_candidate, sim_env, policy_idxs[user_idx])
            if (j % update_cadence == (update_cadence - 1) and j > 0):
                day_in_study = 1 + (j // 2)
                alg_states = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'state.*')
                actions = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'action').flatten()
                pis = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'prob').flatten()
                rewards = get_data_df_values_for_users(data_df, [user_idx], day_in_study, 'reward').flatten()
                alg_candidate.update(alg_states, actions, pis, rewards)
                policy_idxs[user_idx] += 1
                update_idx = int(policy_idxs[user_idx])
                # print("Update Time {} for {}".format(update_idx, user_idx))
                set_update_df_values_for_user(update_df, user_idx, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
            # check if we want to end period of pure exploration for each user
            # for each user's first week, we use the prior
            if alg_candidate.is_pure_exploration_period() and j >= 13:
                alg_candidate.end_pure_exploration_period()

    return data_df, update_df

# returns a int(NUM_USERS / RECRUITMENT_RATE) x RECRUITMENT_RATE array of user indices
# row index represents every other week that they enter the study
def pre_process_users(total_trial_users):
    results = []
    for j, user in enumerate(total_trial_users):
        results.append((int(j), int(2 * (j // RECRUITMENT_RATE)) + 1, user))

    return np.array(results)

### runs experiment with full pooling and incremental recruitment
# users_groups will be a list of tuples where tuple[0] is the user index
# tuple[1] is the week they entered the study, tuple[2] is the user id string
def run_incremental_recruitment_exp(user_groups, alg_candidate, sim_env):
    update_cadence = alg_candidate.get_update_cadence()
    data_df, update_df, _ = create_dfs_full_pooling(user_groups, update_cadence, alg_candidate.get_feature_dim())
    # add in prior values to posterior dataframe
    set_update_df_values(update_df, 0, alg_candidate.posterior_mean, alg_candidate.posterior_var)
    current_groups = user_groups[:RECRUITMENT_RATE]
    update_idx = 0
    week = 1
    while (len(current_groups) > 0):
        print("Week: ", week)
        # do action selection for 14 decision times (7 days)
        num_updates_within_week = int(14 / update_cadence)
        for update_idx_within_week in range(num_updates_within_week):
            for user_tuple in current_groups:
                user_idx, user_entry_date = int(user_tuple[0]), int(user_tuple[1])
                for decision_idx in range(update_cadence):
                    j = (week - user_entry_date) * 14 + (update_idx_within_week * update_cadence) + decision_idx
                    execute_decision_time(data_df, user_idx, j, alg_candidate, sim_env, update_idx)
            ### UPDATE TIME ###
            day_in_study = 1 + (week - 1) * 7 + (update_idx_within_week + decision_idx // 2)
            current_user_idxs = current_groups[:,0].astype(int)
            # update time at the end of each week
            alg_states = get_data_df_values_for_users(data_df, current_user_idxs, day_in_study, 'state.*')
            actions = get_data_df_values_for_users(data_df, current_user_idxs, day_in_study, 'action').flatten()
            pis = get_data_df_values_for_users(data_df, current_user_idxs, day_in_study, 'prob').flatten()
            rewards = get_data_df_values_for_users(data_df, current_user_idxs, day_in_study, 'reward').flatten()
            alg_candidate.update(alg_states, actions, pis, rewards)
            update_idx = 1 + (week - 1) * num_updates_within_week + update_idx_within_week
            # print("UPDATE TIME.", update_idx)
            set_update_df_values(update_df, update_idx, alg_candidate.posterior_mean, alg_candidate.posterior_var)
        # handle adding or removing user groups
        week += 1
        # check if we want to end period of pure exploration
        if alg_candidate.is_pure_exploration_period() and len(current_groups) >= 15:
            alg_candidate.end_pure_exploration_period()
        # biweekly recruitment rate
        if week % 2 != 0:
            # add more users
            # if there are no more users in user_groups to add then current_groups will stay the same
            current_groups = np.concatenate((current_groups, user_groups[np.where(user_groups[:, 1] == str(week))]), axis=0)
            # check if some user group finished the study
            # since we only add users biweekly, users finishing the study should also be
            # at a biweekly cadence
            if (week > TRIAL_LENGTH_IN_WEEKS):
                current_groups = current_groups[RECRUITMENT_RATE:]

    return data_df, update_df
