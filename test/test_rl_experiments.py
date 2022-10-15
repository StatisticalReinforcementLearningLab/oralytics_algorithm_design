from code import rl_experiments
import pandas as pd
import numpy as np
# stack overflow on directory structure to run unit tests:
# https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure

def test_compute_num_updates():
    users_groups = np.array([[0, 0, 'test+0'], [1, 1, 'test+1'], [2, 2, 'test+2']])
    update_cadence = 14

    assert(rl_experiments.compute_num_updates(users_groups, update_cadence) == 11)

# test that experiment data stored in dataframes have the correct format
def test_data_df_sets_value():
    users_groups = np.array([[0, 0, 'test+0'], [1, 1, 'test+1'], [2, 2, 'test+2']])
    update_cadence = 14

    # test construction
    # data_df, update_df, estimating_eqns_df = rl_experiments.create_dfs(users_groups, update_cadence, 9)
    data_df, _, _ = rl_experiments.create_dfs(users_groups, update_cadence, 9)

    # test setting values
    rl_experiments.set_data_df_values_for_user(data_df, '0', 0, 1, 0.75, 120, 120, np.ones(5))
    modified_row = data_df.loc[(data_df['user_idx'] == '0') & (data_df['decision time'] == 0), rl_experiments.FILL_IN_COLS]

    assert(np.array_equal(np.array(modified_row)[0], np.concatenate([[1, 0.75, 120, 120], np.ones(5)])))

def test_data_df_gets_value():
    users_groups = np.array([[0, 0, 'test+0'], [1, 0, 'test+1'], [2, 0, 'test+2']])
    update_cadence = 14

    data_df, _, _ = rl_experiments.create_dfs(users_groups, update_cadence, 9)

    # setting values
    for user_idx in users_groups[:,0]:
        for decision_t in range(14):
            rl_experiments.set_data_df_values_for_user(data_df, user_idx, decision_t, 1, 0.75, 120, 120, np.ones(5))
    # getting reward values
    values = rl_experiments.get_data_df_values_for_users(data_df, users_groups[:,0], 7, 'reward')
    assert(np.array_equal(np.array(values).flatten(), 120. * np.ones(42)))

    # getting prob values
    values = rl_experiments.get_data_df_values_for_users(data_df, users_groups[:,0], 7, 'prob')
    assert(np.array_equal(np.array(values).flatten(), 0.75 * np.ones(42)))

    # getting action values
    values = rl_experiments.get_data_df_values_for_users(data_df, users_groups[:,0], 7, 'action')
    assert(np.array_equal(np.array(values).flatten(), np.ones(42)))

    # getting action values
    values = rl_experiments.get_data_df_values_for_users(data_df, users_groups[:,0], 7, 'state.*')
    assert(np.array_equal(np.array(values), np.ones(shape=(42, 5))))

def test_update_df_sets_value():
    # with incremental recruitment
    users_groups = np.array([[0, 0, 'test+0'], [1, 1, 'test+1'], [2, 2, 'test+2']])
    update_cadence = 14

    _, update_df, _ = rl_experiments.create_dfs(users_groups, update_cadence, 9)

    update_time = 2
    rl_experiments.set_update_df_values(update_df, update_time, np.ones(9), np.ones(shape=(9,9)))
    values = update_df.loc[update_df['update_t'] == update_time]
    assert(np.array_equal(np.array(values).flatten()[1:], np.ones(90)))

def test_estimating_sets_value_eqns_value():
    users_groups = np.array([[0, 0, 'test+0'], [1, 0, 'test+1'], [2, 0, 'test+2']])
    update_cadence = 14

    _, _, estimating_eqns_df = rl_experiments.create_dfs(users_groups, update_cadence, 9)

    update_time = 1
    rl_experiments.set_estimating_eqns_df_values(estimating_eqns_df, update_time, "0", np.zeros(90))
    rl_experiments.set_estimating_eqns_df_values(estimating_eqns_df, update_time, "1", np.ones(90))
    rl_experiments.set_estimating_eqns_df_values(estimating_eqns_df, update_time, "2", 2 * np.ones(90))

    values = np.array(estimating_eqns_df.loc[(estimating_eqns_df['update_t'] == update_time)])[:,3:]
    assert(np.array_equal(values, np.stack([np.zeros(90), np.ones(90), 2 * np.ones(90)])))


test_compute_num_updates()
test_data_df_sets_value()
test_data_df_gets_value()
test_update_df_sets_value()
test_estimating_sets_value_eqns_value()

print("All Tests Passed.")
