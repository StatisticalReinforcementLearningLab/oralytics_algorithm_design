import json
import os
import sys
import read_write_info
import run_experiments

# If this flag is set to True, the jobs won't be submitted to odyssey;
# they will instead be ran one after another in your current terminal
# session. You can use this to either run a sequence of jobs locally
# on your machine, or to run a sequence of jobs one after another
# in an interactive shell on odyssey.
DRYRUN = True
JOB_TYPE = "hyper_plots" #hyper_tuning #simulations #compute_metrics #hyper_plots

# This is the base directory where the results will be stored.
# On Odyssey, you may not want this to be your home directory
# If you're storing lots of files (or storing a lot of data).
OUTPUT_DIR = read_write_info.WRITE_PATH_PREFIX

# This list contains the jobs and simulation enviornments and algorithm
# candidates to search over.
# The list consists of tuples, in which the first element is
# the name of the job (here it describes the exp we want to run)
# and the second is a dictionary of parameters that will be
# be grid-searched over.
# Note that the second parameter must be a dictionary in which each
# value is a list of options.
BASE_ENV_TYPE = ["NON_STAT", "STAT"]
EFFECT_SIZE_SCALES = ["small", "smaller"]
DELAYED_EFFECT_SCALES = ["LOW_R", "MED_R", "HIGH_R"]
# Note: 3396.449 is the noise variance from ROBAS 2 data
# 3412.422 is the noise variance from ROBAS 3 data
# NOISE_VARS = [3396.449, 3412.422]
NOISE_VARS = ["None"]
CLIPPING_VALS = [[0.2, 0.8]]
# COST_PARAMS = [[100, 100]]
### ALGORITHM VARIANTS ###
B_LOGISTICS = [0.515, 5.15]
UPDATE_CADENCES = [2, 14]
CLUSTER_SIZES = ["full_pooling"] # "no_pooling"

COST_PARAMS = [[i,j] for i in range(0, 200, 20) for j in range(0, 200, 20)]

### RUNNING SIMULATIONS ###
# QUEUE = [
#     ('test_v3', dict(sim_env_version=["v3"],
#                        base_env_type=BASE_ENV_TYPE,
#                        effect_size_scale=["None"],
#                        delayed_effect_scale=DELAYED_EFFECT_SCALES,
#                        alg_type=["BLR_AC_V3"],
#                        noise_var=NOISE_VARS,
#                        clipping_vals=CLIPPING_VALS,
#                        b_logistic=B_LOGISTICS,
#                        update_cadence=UPDATE_CADENCES,
#                        cluster_size=CLUSTER_SIZES,
#                        cost_params=COST_PARAMS
#                        )
#     ),
#     ('test_v2', dict(sim_env_version=["v2"],
#                        base_env_type=BASE_ENV_TYPE,
#                        effect_size_scale=EFFECT_SIZE_SCALES,
#                        delayed_effect_scale=DELAYED_EFFECT_SCALES,
#                        alg_type=["BLR_AC_V2"],
#                        noise_var=NOISE_VARS,
#                        clipping_vals=CLIPPING_VALS,
#                        b_logistic=B_LOGISTICS,
#                        update_cadence=UPDATE_CADENCES,
#                        cluster_size=CLUSTER_SIZES,
#                        cost_params=COST_PARAMS
#                        )
#     )
# ]

### HYPERPARAMETER TUNING ###
# QUEUE = [
#     ('hyper_v3', dict(sim_env_version=["v3"],
#                        base_env_type=BASE_ENV_TYPE,
#                        effect_size_scale=["None"],
#                        delayed_effect_scale=DELAYED_EFFECT_SCALES,
#                        alg_type=["BLR_AC_V3"],
#                        cost_params=COST_PARAMS
#                        )
#     ),
#     ('hyper_v2', dict(sim_env_version=["v2"],
#                        base_env_type=BASE_ENV_TYPE,
#                        effect_size_scale=EFFECT_SIZE_SCALES,
#                        delayed_effect_scale=DELAYED_EFFECT_SCALES,
#                        alg_type=["BLR_AC_V2"],
#                        cost_params=COST_PARAMS
#                        )
#     )
# ]

### HYPERPARAMETER PLOTTING ###
QUEUE = [
    ('hyper_v2', dict(sim_env_version=["v2"],
                       base_env_type=BASE_ENV_TYPE,
                       effect_size_scale=EFFECT_SIZE_SCALES,
                       delayed_effect_scale=DELAYED_EFFECT_SCALES
                       )
    ),
    ('hyper_v3', dict(sim_env_version=["v3"],
                       base_env_type=BASE_ENV_TYPE,
                       effect_size_scale=["None"],
                       delayed_effect_scale=DELAYED_EFFECT_SCALES
                       )
    )
]

# what keys in the dictionary do we save the files to
SIM_ENV_NAMES = ["base_env_type", "delayed_effect_scale", "effect_size_scale"]
ALG_CAND_NAMES = ["b_logistic", "update_cadence", "cluster_size"]
OUTPUT_PATH_NAMES = SIM_ENV_NAMES + ALG_CAND_NAMES
HYPER_OUTPUT_PATH_NAMES = SIM_ENV_NAMES + ['cost_params']

def run(exp_dir, exp_name, exp_kwargs):
    '''
    This is the function that will actually execute the job.
    To use it, here's what you need to do:
    1. Create directory 'exp_dir' as a function of 'exp_kwarg'.
       This is so that each set of experiment+hyperparameters get their own directory.
    '''
    if JOB_TYPE == 'simulations' or JOB_TYPE == 'compute_metrics':
        exp_path = os.path.join(exp_dir, "_".join([str(exp_kwargs[key]) for key in OUTPUT_PATH_NAMES]))
    elif JOB_TYPE == 'hyper_tuning':
        exp_path = os.path.join(exp_dir, "_".join([str(exp_kwargs[key]) for key in HYPER_OUTPUT_PATH_NAMES]))
    else:
        exp_path = os.path.join(exp_dir, "_".join([str(exp_kwargs[key]) for key in SIM_ENV_NAMES]))
    print('Results will be stored stored in:', exp_path)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    '''
    2. Run your experiment
    Note: Results are saved after every seed in run_experiments
    '''
    print('Running experiment {}:'.format(exp_name))
    run_experiments.run_experiment(exp_kwargs, exp_path, JOB_TYPE)
    '''
    3. You can find results in 'exp_dir'
    '''
    print('Results are stored in:', exp_path)
    print('with experiment parameters', exp_kwargs)
    print('\n')


def main():
    assert(len(sys.argv) > 2)

    exp_dir = sys.argv[1]
    exp_name = sys.argv[2]
    exp_kwargs = json.loads(sys.argv[3])

    run(exp_dir, exp_name, exp_kwargs)


if __name__ == '__main__':
    main()
