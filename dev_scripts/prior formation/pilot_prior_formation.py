# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# we use sim_env_v3 because a_bar is already normalized (i.e., between [-1, 1]) in that dataset
# but for the original pilot data, a_bar was not normalized (i.e., between [0, 1])
PILOT_DATA = pd.read_csv("https://raw.githubusercontent.com/StatisticalReinforcementLearningLab/oralytics_pilot_data/main/oralytics_pilot_data.csv")
PILOT_USERS = [
    "robas+113@developers.pg.com",
    "robas+27@developers.pg.com",
    "robas+58@developers.pg.com",
    "robas+60@developers.pg.com",
    "robas+64@developers.pg.com",
    "robas+67@developers.pg.com",
    "robas+111@developers.pg.com",
    "robas+110@developers.pg.com",
    "robas+112@developers.pg.com"
    ]
COLS = ['user_id', 'action', 'quality', 'prob', 'state_tod', 'state_b_bar', 'state_a_bar', 'state_app_engage', 'state_bias']
ALG_FEATURES_DF = PILOT_DATA[COLS]
# num. of baseline featuers
D_BASELINE = 5

PILOT_DATA

"""## State Space
---
Baseline:
* 0 - time of day
* 1 - b bar (normalized)
* 2 - a bar (normalized)
* 3 - app engagement
* 4 - bias

Advantage:
* 0 - time of day
* 1 - b bar (normalized)
* 2 - a bar (normalized)
* 3 - app engagement
* 4 - bias

## Helpers
---
"""

# grab user specific df
def get_user_df(user_id):
  return ALG_FEATURES_DF[ALG_FEATURES_DF['user_id'] == user_id]

def get_batch_data(user_id):
  user_df = get_user_df(user_id)
  states_df = user_df.filter(regex='state_*')
  rewards = user_df['quality']
  actions = user_df['action']
  probs = user_df['prob']

  return np.array(states_df), np.array(rewards), np.array(actions), np.array(probs)

def get_all_users_batch_data():
  states_df = ALG_FEATURES_DF.filter(regex='state_*')
  rewards = ALG_FEATURES_DF['quality']

  return np.array(states_df), np.array(rewards)

def get_ols_solution(X, Y):
  # try:
  #   return np.linalg.solve(X.T @ X, X.T @ Y)
  # except:
  #   print('unstable!')
  #   # if matrix is singular then add some noise
  return np.linalg.solve(X.T @ X + 1e-3 * np.diag(np.ones(len(X.T @ X))), X.T @ Y)

all_states, all_rewards = get_all_users_batch_data()
assert(all_states.shape == (len(ALG_FEATURES_DF), D_BASELINE))
assert(all_rewards.shape == (len(ALG_FEATURES_DF),))

"""## Fitting Linear Models
---
1. Linear Model $\theta \in \mathbb{R}^{10}$
2. Linear Model with Action-Centering $\theta \in \mathbb{R}^{15}$
"""

D_LIN = 10
D_LIN_AC = 15

# linear model without action centering
def create_big_phi_no_action_centering(advantage_states, baseline_states, actions):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, actions).T))

  return big_phi

def get_user_lin_model_params(users):
  # fit one theta per user
  thetas = []

  for user_id in users:
    states, rewards, actions, _ = get_batch_data(user_id)
    Phi = create_big_phi_no_action_centering(states, states, actions)
    user_theta = get_ols_solution(Phi, rewards)
    thetas.append(user_theta)

  return np.array(thetas)

# linear model with action centering
def create_big_phi(advantage_states, baseline_states, actions, probs):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, probs).T, \
                       np.multiply(advantage_states.T, (actions - probs)).T,))
  return big_phi

def get_user_lin_ac_model_params(users):
  # fit one theta per user
  thetas = []

  for user_id in users:
    states, rewards, actions, probs = get_batch_data(user_id)
    Phi = create_big_phi(states, states, actions, probs)
    user_theta = get_ols_solution(Phi, rewards)
    thetas.append(user_theta)

  return np.array(thetas)

lin_model_params = get_user_lin_model_params(PILOT_USERS)

ac_model_params = get_user_lin_ac_model_params(PILOT_USERS)

ac_model_params[0]

for val in ["{:.3f}^2 \n".format(sd) for sd in np.std(lin_model_params, axis=0)]:
  print(val)

for val in ["{:.3f} \n".format(sd) for sd in np.std(ac_model_params, axis=0)]:
  print(val)

"""## Fitting $\sigma_n^2$
---
1. We fit one linear regression model per user.
2. We then obtain the weights for each fitted model and calculate residuals.
3. $\sigma_n$ is set to the average SD of the residuals.

Closed Form soluation for linear regression:
$w^* = (X^TX)^{-1}X^Ty$
"""

def calculate_predictions(Phi, rewards):
  theta = get_ols_solution(Phi, rewards)

  return Phi @ theta

def calculate_noise_var_lin_model(users):
  # fit one sigma_n per user to find the variance of sigma_n
  sigma_n_squared_s = []

  for user_id in users:
    states, rewards, actions, _ = get_batch_data(user_id)
    Phi = create_big_phi_no_action_centering(states, states, actions)
    user_predicted_Y = calculate_predictions(Phi, rewards)
    user_residuals = rewards - user_predicted_Y

    sigma_n_squared_s.append(np.var(user_residuals))

  return np.mean(sigma_n_squared_s)

def calculate_noise_var_ac_model(users):
  # fit one sigma_n per user to find the variance of sigma_n
  sigma_n_squared_s = []

  for user_id in users:
    states, rewards, actions, probs = get_batch_data(user_id)
    Phi = create_big_phi(states, states, actions, probs)
    user_predicted_Y = calculate_predictions(Phi, rewards)
    user_residuals = rewards - user_predicted_Y

    sigma_n_squared_s.append(np.var(user_residuals))

  return np.mean(sigma_n_squared_s)

calculate_noise_var_lin_model(PILOT_USERS)

calculate_noise_var_ac_model(PILOT_USERS)

"""## Standardized Effect Sizes
---
"""

def calculate_standard_noise_sd(actual_rewards):
  sample_mean_reward = np.mean(actual_rewards)
  num = np.sum((sample_mean_reward - actual_rewards)**2)
  denom = len(actual_rewards) - 1

  return np.sqrt(num / denom)

def calculate_sample_noise_sd(pred_rewards, actual_rewards, param_dim):
  num = np.sum((pred_rewards - actual_rewards)**2)
  denom = len(actual_rewards) - param_dim

  return np.sqrt(num / denom)

def compute_standard_effect_size(user_params, actual_rewards):
  param_dim = len(user_params)
  standard_noise_sd = calculate_standard_noise_sd(actual_rewards)

  return user_params / standard_noise_sd

def compute_t_test_effect_size(user_params, pred_rewards, actual_rewards):
  param_dim = len(user_params)
  sample_noise_sd = calculate_sample_noise_sd(pred_rewards, actual_rewards, param_dim)

  return user_params / sample_noise_sd

lin_model_params = get_user_lin_model_params(PILOT_USERS)
ac_model_params = get_user_lin_ac_model_params(PILOT_USERS)

def get_standard_effs(lin_model_params, ac_model_params):
  standard_eff_lin = np.zeros(shape=(len(PILOT_USERS),D_LIN))
  standard_eff_ac = np.zeros(shape=(len(PILOT_USERS),D_LIN_AC))
  # standard effect sizes
  for i, user_id in enumerate(PILOT_USERS):
    _, actual_rewards, _, _ = get_batch_data(user_id)
    # linear model
    lin_user_params = lin_model_params[i]
    standard_eff_lin[i] = compute_standard_effect_size(lin_user_params, actual_rewards)
    # AC model
    ac_user_params = ac_model_params[i]
    standard_eff_ac[i] = compute_standard_effect_size(ac_user_params, actual_rewards)

  return standard_eff_lin, standard_eff_ac

# t-test effect sizes
def get_t_test_effs(lin_model_params, ac_model_params):
  t_test_eff_lin = np.zeros(shape=(len(PILOT_USERS),D_LIN))
  t_test_eff_ac = np.zeros(shape=(len(PILOT_USERS),D_LIN_AC))
  # standard effect sizes
  for i, user_id in enumerate(PILOT_USERS):
    states, actual_rewards, actions, probs = get_batch_data(user_id)
    # linear model
    lin_user_params = lin_model_params[i]
    lin_phi = create_big_phi_no_action_centering(states, states, actions)
    lin_pred_rewards = calculate_predictions(lin_phi, actual_rewards)
    t_test_eff_lin[i] = compute_t_test_effect_size(lin_user_params, lin_pred_rewards, actual_rewards)
    # AC model
    ac_user_params = ac_model_params[i]
    ac_phi = create_big_phi(states, states, actions, probs)
    ac_pred_rewards = calculate_predictions(ac_phi, actual_rewards)
    t_test_eff_ac[i] = compute_t_test_effect_size(ac_user_params, ac_pred_rewards, actual_rewards)

  return t_test_eff_lin, t_test_eff_ac

STANDARD_EFF_LIN, STANDARD_EFF_AC = get_standard_effs(lin_model_params, ac_model_params)

T_TEST_EFF_LIN, T_TEST_EFF_AC = get_t_test_effs(lin_model_params, ac_model_params)

"""## Plots"""

# COLORS
# ref: https://htmlcolorcodes.com/colors/shades-of-blue/
SHADES_OF_BLUE = [
    (137, 207, 240),
    (0, 0, 255),
    (135, 206, 235),
    (0, 150, 255),
    (0, 71, 171),
    (111, 143, 175),
    (20, 52, 164),
    (173, 216, 230),
    (31, 81, 255)
]
SHADES_OF_BLUE = [tuple(ti/255 for ti in blue) for blue in SHADES_OF_BLUE]

STATE_NAMES = ['Time of Day', 'Average Past Brushing', 'Average Past Dosage', 'App Engagement', 'Intercept']
LIN_ROW_NAMES = ['Baseline \n', 'Advantage \n']
AC_ROW_NAMES = ['Baseline \n', 'Additional Baseline \n Due To Action-Centering \n', 'Advantage \n']

"""### Matplotlib subplots with row titles
---
"""

def _get_share_ax(share_var, axarr, row, col):
    if share_var=='row':
        if col > 0:
            return axarr[row, col-1]
        return None
    elif share_var=='col':
        if row > 0:
            return axarr[row-1, col]
        return None
    elif share_var and (col>0 or row>0):
        return axarr[0,0]
    return None

def subplots_with_row_titles(nrows, ncols, row_titles=None, row_title_kw=None, sharex=False, sharey=False, subplot_kw=None, grid_spec_kw=None, **fig_kw):
    """
    Creates a figure and array of axes with a title for each row.

    Parameters
    ----------
    nrows, ncols : int
        Number of rows/columns of the subplot grid
    row_titles : list, optional
        List of titles for each row. If included, there must be one title for each row.
    row_title_kw: dict, optional
        Dict with kewords passed to the `~matplotlib.Axis.set_title` function.
        A common use is row_title_kw={'fontsize': 24}
    sharex, sharey : bool or {'none', 'all', 'row', 'col'}, default: False
        Controls sharing of properties among x (*sharex*) or y (*sharey*)
        axes:

        - True or 'all': x- or y-axis will be shared among all subplots.
        - False or 'none': each subplot x- or y-axis will be independent.
        - 'row': each subplot row will share an x- or y-axis.
        - 'col': each subplot column will share an x- or y-axis.

        When subplots have a shared x-axis along a column, only the x tick
        labels of the bottom subplot are created. Similarly, when subplots
        have a shared y-axis along a row, only the y tick labels of the first
        column subplot are created. To later turn other subplots' ticklabels
        on, use `~matplotlib.axes.Axes.tick_params`.
    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.
    gridspec_kw : dict, optional
        Dict with keywords passed to the `~matplotlib.gridspec.GridSpec`
        constructor used to create the grid the subplots are placed on.
    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.
    """
    if row_titles is not None and len(row_titles) != nrows:
        raise ValueError(f'If row_titles is specified, there must be one for each row. Got={row_titles}')
    if subplot_kw is None:
        subplot_kw = {}
    if row_title_kw is None:
        row_title_kw = {}
    if sharex not in {True, False, 'row', 'col'}:
        raise ValueError(f'sharex must be one of [True, False, "row", "col"]. Got={sharex}')
    if sharey not in {True, False, 'row', 'col'}:
        raise ValueError(f'sharey must be one of [True, False, "row", "col"]. Got={sharey}')

    fig, big_axes = plt.subplots(nrows, 1, **fig_kw)
    for (row, big_ax) in enumerate(big_axes):
        if row_titles is not None:
            big_ax.set_title(str(row_titles[row]), **row_title_kw)
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False

    axarr = np.empty((nrows, ncols), dtype='O')
    for row in range(nrows):
        for col in range(ncols):
            sharex_ax = _get_share_ax(sharex, axarr, row, col)
            sharey_ax = _get_share_ax(sharex, axarr, row, col)

            ax= fig.add_subplot(nrows, ncols, row*ncols+col+1,
                                sharex=sharex_ax, sharey=sharey_ax, **subplot_kw)
            axarr[row, col] = ax
    return fig, axarr

rows, cols = 2, 5
fig, axarr = subplots_with_row_titles(rows, cols, figsize=(22,10), #figsize=(cols*8, rows*6),
                                      row_titles=LIN_ROW_NAMES,
                                      row_title_kw=dict(fontsize=24),
                                      sharex='row')
fig.suptitle('Model Parameters For Linear Model', fontweight ="bold", fontsize=25)
x = PILOT_USERS
for row in range(rows):
    for col in range(cols):
        ax = axarr[row, col]
        i = (row + 1) * col
        h = lin_model_params[:, i]
        ax.bar(x, height = h, color = SHADES_OF_BLUE)
        ax.axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
        ax.axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
        ax.set_xticklabels('', rotation='vertical')
        ax.set_title(STATE_NAMES[col], fontsize=20)
        ax.legend(loc='best', fontsize=20)

# adding title per row
fig.supxlabel('Pilot Users', fontsize=21)

fig.tight_layout()
# plt.savefig('session_nums_bar.pdf')
plt.show()

"""### Parameter Plots
---
"""

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(2, num_cols, figsize=(25,10))
fig.suptitle('Parameters For Linear Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN):
  row = i // num_cols
  col = i % num_cols
  h = lin_model_params[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(LIN_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

# adding title per row
fig.supxlabel('Pilot Users', fontsize=21)

fig.tight_layout()
plt.savefig('lin_model_params.pdf')
plt.show()

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(3, num_cols, figsize=(25,15))
fig.suptitle('Parameters For Action-Centering Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN_AC):
  row = i // num_cols
  col = i % num_cols
  h = ac_model_params[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  ax[row, col].set_xlabel('Pilot Users', fontsize=18) # set x label
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(AC_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

fig.tight_layout()
plt.savefig('ac_model_params.pdf')
plt.show()

"""### Effect Size Plots
---

#### Linear Model
---
"""

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(2, num_cols, figsize=(25,10))
fig.suptitle('Standard Effect Sizes For Linear Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN):
  row = i // num_cols
  col = i % num_cols
  h = STANDARD_EFF_LIN[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(LIN_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

# adding title per row
fig.supxlabel('Pilot Users', fontsize=21)

fig.tight_layout()
plt.savefig('lin_model_standard_eff.pdf')
plt.show()

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(2, num_cols, figsize=(25,10))
fig.suptitle('T-Test Effect Sizes For Linear Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN):
  row = i // num_cols
  col = i % num_cols
  h = T_TEST_EFF_LIN[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(LIN_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

# adding title per row
fig.supxlabel('Pilot Users', fontsize=21)

fig.tight_layout()
plt.savefig('lin_model_t_test_eff.pdf')
plt.show()

"""#### AC Model
---
"""

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(3, num_cols, figsize=(25,15))
fig.suptitle('Standard Effect Sizes For Action-Centering Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN_AC):
  row = i // num_cols
  col = i % num_cols
  h = STANDARD_EFF_AC[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  ax[row, col].set_xlabel('Pilot Users', fontsize=18) # set x label
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(AC_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

fig.tight_layout()
plt.savefig('ac_model_standard_eff.pdf')
plt.show()

num_cols = 5
x = PILOT_USERS
x_tick_names = [user.split("@")[0] for user in PILOT_USERS]
fig, ax = plt.subplots(3, num_cols, figsize=(25,15))
fig.suptitle('T-Test Effect Sizes For Action-Centering Model \n', fontweight ="bold", fontsize=25)
for i in range(D_LIN_AC):
  row = i // num_cols
  col = i % num_cols
  h = T_TEST_EFF_AC[:, i]

  ax[row, col].bar(x, height = h, color = SHADES_OF_BLUE)
  ax[row, col].axhline(y=np.mean(h), color='r', linestyle='-', label='Mean: {:.3f}'.format(np.mean(h)))
  ax[row, col].axhline(y=np.median(h), color='y', linestyle='-', label='Median: {:.3f}'.format(np.median(h)))
  ax[row, col].set_xticklabels('', rotation='vertical')
  ax[row, col].set_title(STATE_NAMES[col], fontsize=20) # set title
  ax[row, col].set_xlabel('Pilot Users', fontsize=18) # set x label
  # if first graph in the row
  if col == 0:
    ax[row, col].set_ylabel(AC_ROW_NAMES[row], fontsize=20) # set y label
  ax[row, col].legend(loc='best', fontsize=20)

fig.tight_layout()
plt.savefig('ac_model_t_test_eff.pdf')
plt.show()

"""## Simulating Action-Selection Probs
---
"""

import scipy.stats as stats

all_states, all_rewards = get_all_users_batch_data()
assert(all_states.shape == (len(ALG_FEATURES_DF), D_BASELINE))
assert(all_rewards.shape == (len(ALG_FEATURES_DF),))

SIGMA_N_2 = 3878
ALPHA_0_MU = [18, 0, 30, 0, 73]
BETA_MU = [0, 0, 0, 53, 0]
PRIOR_MU = np.array(ALPHA_0_MU + BETA_MU + BETA_MU)
ALPHA_0_SIGMA = [73**2, 25**2, 95**2, 27**2, 83**2]
BETA_SIGMA = [12**2, 33**2, 35**2, 56**2, 17**2]
PRIOR_SIGMA = np.diag(np.array(ALPHA_0_SIGMA + BETA_SIGMA + BETA_SIGMA))

D_ADVANTAGE = 5

from scipy.special import expit

# generalized logistic function https://en.wikipedia.org/wiki/Generalised_logistic_function
# lower and upper asymptotes (clipping values)
L_min = 0.2
L_max = 0.8
# larger values of b > 0 makes curve more "steep"
B_logistic = 0.515
# larger values of c > 0 shifts the value of function(0) to the right
C_logistic = 3
# larger values of k > 0 makes the asmptote towards upper clipping less steep
# and the asymptote towards the lower clipping more steep
K_logistic = 1

# uses scipy.special.expit for numerical stability
def stable_generalized_logistic(x):
    num = L_max - L_min
    stable_exp = expit(B_logistic * x - np.log(C_logistic))
    stable_exp_k = stable_exp**K_logistic

    return L_min + num * stable_exp_k

def bayes_lr_action_selector(beta_post_mean, beta_post_var, advantage_state):
  # using the genearlized_logistic_func, probabilities are already clipped to asymptotes
  mu = advantage_state @ beta_post_mean
  std = np.sqrt(advantage_state @ beta_post_var @ advantage_state.T)
  posterior_prob = stats.norm.expect(func=stable_generalized_logistic, loc=mu, scale=std)

  return posterior_prob

SIMULATED_PROBS = []
for state in all_states:
  adv_state = state
  prob = bayes_lr_action_selector(PRIOR_MU[-D_ADVANTAGE:], PRIOR_SIGMA[-D_ADVANTAGE:,-D_ADVANTAGE:], adv_state)
  SIMULATED_PROBS.append(prob)

N = len(SIMULATED_PROBS)
n_bins = int(np.ceil(np.sqrt(N))) if int(np.ceil(np.sqrt(N))) > 5 else 5
data = SIMULATED_PROBS
data_mean = np.mean(data)

plt.hist(data, n_bins, histtype ='bar')

# plot the true mean
plt.axvline(x=data_mean, color='red', linestyle='dotted', label='mean:{:.3f}'.format(data_mean))
plt.legend(loc='best') # plot legend plt.legend(prop ={'size': 10})
plt.xlabel('Action-Selection Probabilities', fontsize=12) # set x label
plt.ylabel('Frequency', fontsize=12) # set y label

plt.title('Simulated Probabilities With Prior', fontsize=15) #, fontweight ="bold")
file_name = 'simulated_probs_from_prior'
plt.savefig("{}.pdf".format(file_name), bbox_inches="tight")

plt.show()

