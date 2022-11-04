#!/bin/bash

for SIM_ENV in STAT_LOW_R STAT_MED_R STAT_HIGH_R NON_STAT_LOW_R NON_STAT_MED_R NON_STAT_HIGH_R
do
  export SIM_ENV
  for CLIPPING_VALS in 0.1_0.9 0.35_0.75
  do
    export CLIPPING_VALS
    for B_LOGISTIC in 0.515 0.343
    do
      export B_LOGISTIC
        sbatch --job-name=job_${SIM_ENV}_${CLIPPING_VALS}_${B_LOGISTIC} \
        -o outs_and_errs/%j_out.txt -e outs_and_errs/%j_errs.txt \
          src/run_rl_experiments.sbatch
        sleep 0.01
    done
  done
done
