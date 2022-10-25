# Naming Convention For Data and Algorithm Statistics

This document explains the column names and data frames that are produced with each experiment trial. After running a simulated experiment trial, the experiment will produce: 1) a data dataframe, 2) an update dataframe with posterior parameteters, 3) an estimating equations dataframe and 4) a policy gradients dataframe.

## Data DataFrame
* `user_idx`: unique trial index for the user, starts with 0
* `user_id`: ROBAS 3 user id used to obtain environment parameters (user_id can repeat so that's why we have a unique `user_idx`)
* `user_entry_decision_t`: first calendar decision time that the user enters the study
* `user_last_decision_t`: last calendar decision time for the user
*  `user_decision_t`: indexes the user-specific decision time starts with 0, ends with 139 (note: even values denote the morning decision time, odd values denote the evening decision time)
* `calendar_decision_t`: calendar decision time, starts with 0
* `day_in_study`: calendar day in study, starts with 1
* `policy_idx`: indexes which policy was used for action selection, this value is congruent with `update_t` in the following sections
* `action`: 1 or 0. 1 signifies sending a message, 0 signifies not sending a message
* `prob`: posterior probability of selecting action 1 for that decision time
* `reward`: reward, a function of brushing quality and cost of sending an intervention, for that decision time (used to train RL algorithm)
* `quality`: actual brushing quality for that decision time (used to evaluate algorithm)
* `state.{}`: flattened state (context) vector observed by the algorithm at decision time and used for training

## Update DataFrame
* `update_t`: index for the update time and the policy, starts with 0. 0 index refers to the prior distribution before any data update and 1 index refers to the first posterior update using data.
* `posterior_mu.{}`: flattened posterior mean vector where `{}` indexes into the vector, starts with 0
* `posterior_var.{}.{}`: flattened posterior covariance matrix where `{}` indexes the row and the second `{}` indexes the column, starts with 0

## Estimating Equations DataFrame
* `update_t`: index for the update time and the policy, starts with 1
* `user_idx`: unique trial index for the user, starts with 0
* `user_id`: ROBAS 3 user id used to obtain environment parameters (user_id can repeat so that's why we have a unique `user_idx`)
* `mean_estimate.{}`: flattened mean vector where `{}` indexes into the vector, starts with 0
* `var_estimate.{}.{}`: flattened covariance matrix where `{}` indexes the row and the second `{}` indexes the column, starts with 0

## Policy Gradients DataFrame
In progress...
