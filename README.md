# Oralytics Algorithm Design Decisions

This repository contains code for running simulations to make design decisions for the Oralytics RL algorithm. 

### Simulation Environment
* Fitting parameters for the simualtion environment base model: [fitting_simulation_base_model.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/code/fitting_environment_models.py)
* Selecting best environment model for each user: [environment_base_model_selection.py](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/code/environment_base_model_selection.py)

1. `python3 fitting_environment_models.py` will fit each ROBAS 3 user to a base model class (hurdle with a square root transform model or zero-inflated poisson model) and save parameters associated with each model to csv files. 

2. `python3 environment_base_model_selection.py` will take the csv files generated as described above in step 1 and will compute the RMSE of the model and the observed ROBAS 3 data and choose the base model class that yields a lower RMSE for that user. Outputs: [stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/stat_user_models.csv) and [non_stat_user_models.csv](https://github.com/StatisticalReinforcementLearningLab/oralytics_algorithm_design/blob/main/sim_env_data/non_stat_user_models.csv)

### Running Unit Tests
To run tests, you need to be in the root folder and then run for example `python3 -m unittest test.test_rl_experiments` if you want to run the `test_rl_experiments.py` file in the test folder.
