#!/bin/bash

for SIM_ENV in STAT_LOW_R STAT_MED_R STAT_HIGH_R NON_STAT_LOW_R NON_STAT_MED_R NON_STAT_HIGH_R
do
  export SIM_ENV
  for EFFECT_SIZE_SCALE in small smaller
  do
    export EFFECT_SIZE_SCALE
    for x1 in {0..180..20}
    do
      export x1
      for x2 in {0..180..20}
      do
        export x2
        export CLIPPING_VALS=0.2_0.8
        export B_LOGISTIC=0.515
        export ALG_TYPE=BLR_AC
        export UPDATE_CADENCE=2
        export CLUSTER_SIZE=72
        export TUNING_HYPERS=True
        sbatch --job-name=job_${SIM_ENV}_${EFFECT_SIZE_SCALE}_${x1}_${x2} \
        -o outs_and_errs/%j_out.txt -e outs_and_errs/%j_errs.txt \
        src/run_rl_experiments.sbatch
        sleep 0.01
      done
    done
  done
done
