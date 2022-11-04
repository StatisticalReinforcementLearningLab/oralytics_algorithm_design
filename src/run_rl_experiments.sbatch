#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=5000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o outs_and_errs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e outs_and_errs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid

module load Anaconda/5.0.1-fasrc02

source activate test_env

python3 src/run_rl_experiment.py --sim_env_type=${SIM_ENV} --clipping_vals=${CLIPPING_VALS} --b_logistic=${B_LOGISTIC}

# python3 src/run_rl_experiment.py --sim_env_type='STAT_LOW_R' --clipping_vals='0.1_0.9' --b_logistic=0.515
