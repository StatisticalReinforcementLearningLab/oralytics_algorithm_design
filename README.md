# Oralytics Algorithm Design Decisions

This repository contains code for running simulations to make design decisions for the Oralytics RL algorithm. Full details can be found in Appendix A of the [Oralytics protocol paper](https://www.sciencedirect.com/science/article/abs/pii/S1551714424000387).

### Simulation Environment
To make the final design decisions on the Oralytics algorithm for the main study of Oralytics, we used the Oralytics V2 simulation environment test bed to run simulations. V2 was built off of ROBAS 3 data and included simulated app opening behavior. Although not used in the final design phase, we include code for V1, a previous (now deprecated) version and V3, a version built off of Oralytics pilot data that was not needed for our simulations, but could still be useful for other research teams.

Code for building and evaluating teh V2 simulation test bed:
* Fitting parameters for the simualtion environment base model: [v1v2_fitting_environment_models.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/sim_env_v1v2/v1v2_fitting_environment_models.py)
* Selecting best environment model for each user: [v1v2_env_base_model_selection.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/sim_env_v1v2/v1v2_env_base_model_selection.py)
* Calculating realistic effect sizes to impute: [calculate_effect_sizes.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/sim_env_v1v2/calculate_effect_sizes.py)
* Computing statistics and checking how reasonable imputed effect sizes are: [effect_size_check.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/sim_env_v1v2/effect_size_check.py)
* Calculating a population-level app opening probability using Oralytics pilot data: [app_opening_prob_calculation.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/sim_env_v3/app_opening_prob_calculation.py)

We describe the procedure for creating the V2 simulation environment test bed:
1. Running `python3 dev_scripts/sim_env_v1v2/v1v2_fitting_environment_models.py` will fit each ROBAS 3 user to a base model class (hurdle with a square root transform model or zero-inflated poisson model) and save parameters associated with each model to csv files. 

2. Running `python3 dev_scripts/sim_env_v1v2/v1v2_env_base_model_selection.py` will take the csv files generated as described above in step 1 and will compute the RMSE of the model and the observed ROBAS 3 data and choose the base model class that yields a lower RMSE for that user. Outputs: [stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/stat_user_models.csv) and [non_stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/non_stat_user_models.csv)

3. Running `python3 dev_scripts/sim_env_v1v2/calculate_effect_sizes.py` will take in [stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/stat_user_models.csv) and [non_stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/non_stat_user_models.csv) and calculate effect sizes to impute based on each environment's fitted parameters. Outputs are in the [sim_env_data](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/tree/main/sim_env_data) folder with suffix pattern `effect_sizes.p`.

4. Running `python3 dev_scripts/sim_env_v1v2/calculate_effect_sizes.py` will take in [stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/stat_user_models.csv), [non_stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/non_stat_user_models.csv), and ROBAS 3 raw data and calculates standard effect sizes to compare with the imputed effect sizes created in step 3.

5. Running `python3 dev_scripts/sim_env_v3/app_opening_prob_calculation.py` will take in Oralytics pilot data and calculate app opening probabilites for each participant in the pilot study. This value was used to determine the population-level app opening probability we imputed in the V2 test bed.

### Running Experiments
To run experiments:
1. Fill in the read and write path in [read_write_info.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/src/read_write_info.py). This specifies what path to read data from and what path to write results to. 
2. In [run.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/src/run.py), specify experiments parameters as instructed in the file. Example: specify the simulation environment variants and algorithm candidate properties. You must modify the `JOB_TYPE` field to specify what job to run. In addition, you must modify the `DRYRUN` field to specify running jobs in parallel or sequentially. `DRYRUN = True` runs jobs one after the other (this is a good practice to test out new code initially). Switch to `DRYRUN = False` to run experiments in parallel.  

There are 4 types of jobs:
  * `simulations`: runs the main set of experiments for each algorithm candidate in each simulation environment variant
  * `compute_metrics`: computes desired metrics using outputs from `simulations`
  * `hyper_tuning`: runs the hyperparameter tuning for the reward parameters after finalizing the algorithm candidate 
  * `hyper_plots`: computes hyperparameter grid plots using outputs from `hyper_tuning`
    
For each job type, there are commented out example lists called `QUEUE`. Please comment / uncomment to fit your job type but make sure only one `QUEUE` is uncommented at a time.

3. Run `python3 src/submit_batch` on the cluster to submit jobs and run in parallel.

### Fitting the Prior
For the Oralytics main study, we designed the prior based off of Oralytics pilot data and in discussion with domain experts. [pilot_prior_formation.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/dev_scripts/prior%20formation/pilot_prior_formation.py) is the code for calculating the statistics and plots which informed the design of the prior. For knowledge sharing, we have kept additional scripts in the folder [prior formation](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/tree/main/dev_scripts/prior%20formation) which contains code using GEE analysis as a measure for significance testing.

### Running Unit Tests
To run tests, you need to be in the root folder and then run for example `python3 -m unittest test.test_rl_experiments` if you want to run the `test_rl_experiments.py` file in the test folder.


