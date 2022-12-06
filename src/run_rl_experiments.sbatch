#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outs_and_errs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e outs_and_errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda/5.0.1-fasrc02

source activate test_env

python3 src/run_rl_experiment.py --sim_env_type=${SIM_ENV} --alg_type=${ALG_TYPE} \
--b_logistic=${B_LOGISTIC} --clipping_vals=${CLIPPING_VALS} \
--update_cadence=${UPDATE_CADENCE} --cluster_size=${CLUSTER_SIZE} \
--effect_size_scale=${EFFECT_SIZE_SCALE} --algorithm_val_1=${x1} --algorithm_val_2=${x2} \
--tuning_hypers=${TUNING_HYPERS}

# python3 src/run_rl_experiment.py --sim_env_type='NON_STAT_LOW_R' --alg_type='BLR_AC' \
# --b_logistic="0.515" --clipping_vals='0.2_0.8' \
# --update_cadence="2" --cluster_size="1" --effect_size_scale="smaller"

# python3 src/run_rl_experiment.py --sim_env_type='NON_STAT_LOW_R' --alg_type='BLR_AC' \
# --b_logistic="0.515" --clipping_vals='0.2_0.8' \
# --update_cadence="2" --cluster_size="72" --effect_size_scale="smaller" \
# --algorithm_val_1=100 --algorithm_val_2=100 --tuning_hypers=True
