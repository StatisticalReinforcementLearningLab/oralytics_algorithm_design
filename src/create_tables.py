import pandas as pd
import pickle
import itertools
import read_write_info
import run
import experiment_global_vars
import os

#### GENERATING TABLE FOR v2 ####
BASE_ENV_TYPE = ["STAT", "NON_STAT"]
DELAYED_EFFECT_SCALES = ["LOW_R", "MED_R", "HIGH_R"]
V2_EXP_NAME = run.QUEUE[1][0]
ENV_NAMES_SMALLER = ["{}_{}_smaller".format(*env_params) for env_params in itertools.product(*[BASE_ENV_TYPE, DELAYED_EFFECT_SCALES])]
ENV_NAMES_SMALL = ["{}_{}_small".format(*env_params) for env_params in itertools.product(*[BASE_ENV_TYPE, DELAYED_EFFECT_SCALES])]

#### GENERATING TABLE FOR v3 ####
ENV_NAMES_V3 = ["{}_{}_None".format(*env_params) for env_params in itertools.product(*[BASE_ENV_TYPE, DELAYED_EFFECT_SCALES])]
V3_EXP_NAME = run.QUEUE[0][0]

ALG_CANDIDATES = dict(
    B_LOGISTICS=[0.515, 5.15],
    UPDATE_CADENCE=[14, 2],
    CLUSTER_SIZE=["full_pooling", "no_pooling"]
)
ALG_NAMES = ["{}_{}_{}".format(*candidate_params) for candidate_params in itertools.product(*list(ALG_CANDIDATES.values()))]
ALG_TABLE_NAMES = ["B = 0.515, Weekly, Full Pooling", "B = 0.515, Weekly, No Pooling", "B = 0.515, Daily, Full Pooling", \
                    "B = 0.515, Daily, No Pooling", "B = 5.15, Weekly, Full Pooling", "B = 5.15, Weekly, No Pooling", \
                    "B = 5.15, Daily, Full Pooling", "B = 5.15, Daily, No Pooling"]

### x axis is simulation environments ###
def print_tables(exp_name, env_names, alg_names, alg_table_names):
    print("FOR: ", exp_name)
    all_avg_qualities = []
    all_lower_25_qualities = []

    for SIM_ENV in env_names:
        sim_env_avg_qualities = []
        sim_env_lower_25_qualities = []
        for ALG_NAME in alg_names:
            print("FOR {} {}".format(ALG_NAME, SIM_ENV))
            exp_dir = os.path.join(run.OUTPUT_DIR, exp_name)
            exp_path = os.path.join(exp_dir, "{}_{}".format(SIM_ENV, ALG_NAME))
            print("OPENING IN THIS PATH: ", exp_path)
            avg_pickle_location = exp_path + '/avg.p'
            lower_25_pickle_location = exp_path + '/low_25.p'
            with open(avg_pickle_location, 'rb') as handle:
                env_alg_mean = pickle.load(handle)
            with open(lower_25_pickle_location, 'rb') as handle:
                env_alg_lower_25 = pickle.load(handle)
            sim_env_avg_qualities.append(env_alg_mean)
            sim_env_lower_25_qualities.append(env_alg_lower_25)

        all_avg_qualities.append(sim_env_avg_qualities)
        all_lower_25_qualities.append(sim_env_lower_25_qualities)

    # formatting metrics into df and then convert to latex
    total_avg_vals = dict(ALG_CANDS=alg_table_names)
    avg_vals = {env_names[i]: all_avg_qualities[i] for i in range(len(env_names))}
    total_avg_vals.update(avg_vals)
    df_avg_qualities = pd.DataFrame(total_avg_vals)

    total_lower_25_vals = dict(ALG_CANDS=alg_table_names)
    lower_25_vals = {env_names[i]: all_lower_25_qualities[i] for i in range(len(env_names))}
    total_lower_25_vals.update(lower_25_vals)
    df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

    df_avg_qualities.style.highlight_max(axis=None, props='bfseries: ;')
    df_lower_25_qualities.style.highlight_max(axis=None, props='bfseries: ;')

    print(df_avg_qualities.to_latex(index=False))
    print(df_lower_25_qualities.to_latex(index=False))

print_tables(V2_EXP_NAME, ENV_NAMES_SMALLER, ALG_NAMES, ALG_TABLE_NAMES)
print_tables(V2_EXP_NAME, ENV_NAMES_SMALL, ALG_NAMES, ALG_TABLE_NAMES)
print_tables(V3_EXP_NAME, ENV_NAMES_V3, ALG_NAMES, ALG_TABLE_NAMES)

# all_avg_qualities = []
# all_lower_25_qualities = []
#
# for SIM_ENV in ENV_NAMES_SMALLER:
#     sim_env_avg_qualities = []
#     sim_env_lower_25_qualities = []
#     for ALG_NAME in ALG_NAMES:
#         print("FOR {} {}".format(ALG_NAME, SIM_ENV))
#         exp_dir = os.path.join(run.OUTPUT_DIR, V2_EXP_NAME)
#         exp_path = os.path.join(exp_dir, "{}_{}".format(SIM_ENV, ALG_NAME))
#         print("OPENING IN THIS PATH: ", exp_path)
#         avg_pickle_location = exp_path + '/avg.p'
#         lower_25_pickle_location = exp_path + '/low_25.p'
#         with open(avg_pickle_location, 'rb') as handle:
#             env_alg_mean = pickle.load(handle)
#         with open(lower_25_pickle_location, 'rb') as handle:
#             env_alg_lower_25 = pickle.load(handle)
#         sim_env_avg_qualities.append(env_alg_mean)
#         sim_env_lower_25_qualities.append(env_alg_lower_25)
#
#     all_avg_qualities.append(sim_env_avg_qualities)
#     all_lower_25_qualities.append(sim_env_lower_25_qualities)
#
# # formatting metrics into df and then convert to latex
# total_avg_vals = dict(ALG_CANDS=ALG_NAMES)
# avg_vals = {ENV_TABLE_NAMES[i]: all_avg_qualities[i] for i in range(len(ENV_TABLE_NAMES))}
# total_avg_vals.update(avg_vals)
# df_avg_qualities = pd.DataFrame(total_avg_vals)
#
# total_lower_25_vals = dict(ALG_CANDS=ALG_NAMES)
# lower_25_vals = {ENV_TABLE_NAMES[i]: all_lower_25_qualities[i] for i in range(len(ENV_TABLE_NAMES))}
# total_lower_25_vals.update(lower_25_vals)
# df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)
#
# print(df_avg_qualities.to_latex(index=False))
# print(df_lower_25_qualities.to_latex(index=False))

### x axis is algorithm candidates ###
# all_avg_qualities = []
# all_lower_25_qualities = []
#
# for ALG_NAME in ALG_NAMES:
#     alg_avg_qualities = []
#     alg_lower_25_qualities = []
#     for SIM_ENV in ENV_NAMES:
#         print("FOR {} {}".format(ALG_NAME, SIM_ENV))
#         string_prefix = READ_PATH_PREFIX + "{}_{}".format(SIM_ENV, ALG_NAME)
#         alg_qualities = format_qualities(string_prefix)
#         alg_avg_qualities.append(report_mean_quality(alg_qualities))
#         alg_lower_25_qualities.append(report_lower_25_quality(alg_qualities))
#
#     all_avg_qualities.append(alg_avg_qualities)
#     all_lower_25_qualities.append(alg_lower_25_qualities)

# formatting metrics into df and then convert to latex
# total_avg_vals = dict(SIM_ENV=ENV_NAMES)
# avg_vals = {ALG_NAMES[i]: all_avg_qualities[i] for i in range(len(ALG_NAMES))}
# total_avg_vals.update(avg_vals)
# df_avg_qualities = pd.DataFrame(total_avg_vals)
#
# total_lower_25_vals = dict(SIM_ENV=ENV_NAMES)
# lower_25_vals = {ALG_NAMES[i]: all_lower_25_qualities[i] for i in range(len(ALG_NAMES))}
# total_lower_25_vals.update(lower_25_vals)
# df_lower_25_qualities = pd.DataFrame(total_lower_25_vals)

# print(df_avg_qualities.to_latex(index=False))
# print(df_lower_25_qualities.to_latex(index=False))
