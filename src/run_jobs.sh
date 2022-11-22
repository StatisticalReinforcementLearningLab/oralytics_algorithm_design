#!/bin/bash

for SIM_ENV in STAT_LOW_R STAT_MED_R STAT_HIGH_R NON_STAT_LOW_R NON_STAT_MED_R NON_STAT_HIGH_R
do
  export SIM_ENV
  for CLIPPING_VALS in 0.2_0.8
  do
    export CLIPPING_VALS
    for B_LOGISTIC in 0.515 5.15
    do
      export B_LOGISTIC
      for ALG_TYPE in BLR_AC
      do
        export ALG_TYPE
        for UPDATE_CADENCE in 14 2
        do
          export UPDATE_CADENCE
          for CLUSTER_SIZE in 72 1
          do
            export CLUSTER_SIZE
            sbatch --job-name=job_${SIM_ENV}_${ALG_TYPE}_${B_LOGISTIC}_${CLIPPING_VALS}_${UPDATE_CADENCE}_${CLUSTER_SIZE} \
            -o outs_and_errs/%j_out.txt -e outs_and_errs/%j_errs.txt \
            src/run_rl_experiments.sbatch
            sleep 0.01
          done
        done
      done
    done
  done
done
