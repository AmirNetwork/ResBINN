

# Author: Amir Ghorbani (amirg@unimelb.edu.au)

import os
import math
import random
import pickle
import warnings
from time import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy                              # top-level SciPy
import scipy.stats as stats               # full stats namespace
from  scipy.stats import norm, uniform

import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize

import collections
import collections.abc                    # for the Iterable alias below

# ─── TensorFlow / Keras & TF-Lattice ────────────────────────────────────────
import tensorflow as tf
import tensorflow_lattice as tfl
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    MultiHeadAttention, LayerNormalization, Activation, Input, Dense, Reshape,
    Concatenate, Layer, Dropout, BatchNormalization, Embedding, Flatten,
    LeakyReLU, ReLU, Lambda
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop, Adam

# Keras wrappers
from scikeras.wrappers import KerasRegressor, KerasClassifier

# ─── PyTorch ────────────────────────────────────────────────────────────────
import torch
from   torch import tensor
import torch.nn.functional as F
import torch.distributions.log_normal  as log_normal
import torch.distributions.bernoulli   as bernoulli

# ─── PyLogit & companions ───────────────────────────────────────────────────
import pylogit as pl
import lxml                              # dependency for PyLogit XML parsing
import statsmodels.api as sm
from   statsmodels.formula.api import logit

# ─── Scikit-learn stack ─────────────────────────────────────────────────────
from sklearn.base        import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.inspection  import PartialDependenceDisplay, partial_dependence
from sklearn.metrics     import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error, brier_score_loss
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.multiclass import unique_labels

# --------------------------- 
collections.Iterable = collections.abc.Iterable  # legacy fix for PyLogit
warnings.filterwarnings("ignore")
# ------------------------


WTP_MIS_SPEC = float(os.getenv("WTP_MIS_SPEC", "0"))   

def _mis_spec(val):
    if WTP_MIS_SPEC == 0:
        return val
    # draw ε ~ N(−σ²/2, σ²) so that E[exp(ε)] = 1
    mu = -0.5 * WTP_MIS_SPEC**2
    eps = _rng_wtp.normal(mu, WTP_MIS_SPEC)
    return val * np.exp(eps)


SIGMA_VALUES = [ 0.05, 0.10, 0.25, 0.50]


num_experiments  = len(SIGMA_VALUES)              
print("Σ sweep:", SIGMA_VALUES)                    


def brier_multi(targets, probs):
    return np.mean(np.sum((probs - targets) ** 2, axis=1))


def save_clipboard(X):
    pd.DataFrame(X).to_clipboard(index=False, header=False)


np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None  


def compute_VOT_POP(X, Y):
    VOT_temp = []
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            VOT_temp.append(X[i] / Y[j])
    return np.array(VOT_temp)


def compute_VOT_IND(X, Y):
    VOT_temp = []
    for i in range(X.shape[0]):
        VOT_temp.append(np.median(compute_VOT_POP(X[i, :], Y[i, :])))
    return np.array(VOT_temp)





warnings.filterwarnings('ignore')



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
file_path = os.path.join(downloads_path, 'swissmetro.dat')

try:
    df_wide = pd.read_csv(file_path, sep='\t')
    print("Data loaded successfully. First few rows:")
    print(df_wide.head())  
except Exception as e:
    print(f"Error reading the file: {e}")

df_wide = df_wide[
    (df_wide.AGE != 6) & (df_wide.CHOICE != 0) & (df_wide.INCOME != 4) & (df_wide.PURPOSE != 9) & (df_wide.CAR_AV != 0)]


df_wide["AGES"] = 0
df_wide.loc[df_wide["AGE"] == 1, "AGES"] = 1
df_wide.loc[df_wide["AGE"] == 2, "AGES"] = 2
df_wide.loc[df_wide["AGE"] == 3, "AGES"] = 3
df_wide.loc[df_wide["AGE"] == 4, "AGES"] = 4
df_wide.loc[df_wide["AGE"] == 5, "AGES"] = 5


df_wide["INCOMES"] = 0
df_wide.loc[df_wide["INCOME"].isin([0, 1]), "INCOMES"] = 1
df_wide.loc[df_wide["INCOME"] == 2, "INCOMES"] = 2
df_wide.loc[df_wide["INCOME"] == 3, "INCOMES"] = 3
df_wide.loc[df_wide["INCOME"] == 4, "INCOMES"] = 4


df_wide["PURPOSES"] = 0
df_wide.loc[df_wide["PURPOSE"].isin([1, 5]), "PURPOSES"] = 1
df_wide.loc[df_wide["PURPOSE"].isin([2, 6]), "PURPOSES"] = 2
df_wide.loc[df_wide["PURPOSE"].isin([3, 7]), "PURPOSES"] = 3
df_wide.loc[df_wide["PURPOSE"].isin([4, 8]), "PURPOSES"] = 4


df_wide["TRAIN_TT"] = df_wide["TRAIN_TT"] / 100.0
df_wide["SM_TT"] = df_wide["SM_TT"] / 100.0
df_wide["CAR_TT"] = df_wide["CAR_TT"] / 100.0


df_wide["TRAIN_HE"] = df_wide["TRAIN_HE"] / 100.0
df_wide["SM_HE"] = df_wide["SM_HE"] / 100.0

df_wide["TRAIN_TC"] = df_wide["TRAIN_CO"] / 100.0
df_wide["SM_TC"] = df_wide["SM_CO"] / 100.0
df_wide["CAR_TC"] = df_wide["CAR_CO"] / 100.0

df_wide.loc[(df_wide["GA"] == 1), "TRAIN_TC"] = 0
df_wide.loc[(df_wide["GA"] == 1), "SM_TC"] = 0


df_wide["AGE_2"] = (df_wide["AGE"] == 2).astype(int)
df_wide["AGE_3"] = (df_wide["AGE"] == 3).astype(int)
df_wide["AGE_4"] = (df_wide["AGE"] == 4).astype(int)
df_wide["AGE_5"] = (df_wide["AGE"] == 5).astype(int)

df_wide["INCOME_2"] = (df_wide["INCOME"] == 2).astype(int)
df_wide["INCOME_3"] = (df_wide["INCOME"] == 3).astype(int)
df_wide["INCOME_4"] = (df_wide["INCOME"] == 4).astype(int)

df_wide["PURPOSE_2"] = (df_wide["PURPOSE"].isin([2, 6])).astype(int)
df_wide["PURPOSE_3"] = (df_wide["PURPOSE"].isin([3, 7])).astype(int)
df_wide["PURPOSE_4"] = (df_wide["PURPOSE"].isin([4, 8])).astype(int)

df_wide["LUGGAGE_1"] = (df_wide["LUGGAGE"] == 1).astype(int)
df_wide["LUGGAGE_3"] = (df_wide["LUGGAGE"] == 3).astype(int)

df_wide["TRAIN_CLASS"] = 1 - df_wide["FIRST"]

print("Number of unique individuals:", df_wide['ID'].nunique())


ind_variables = ['ID', 'GA', 'AGE_2', 'AGE_3', 'AGE_4', 'AGE_5', 'INCOME_2', 'INCOME_3', 'INCOME_4',
                 'PURPOSE_2', 'PURPOSE_3', 'PURPOSE_4', 'AGES', 'INCOMES', 'PURPOSES']

alt_varying_variables = {
    'TT': {1: 'TRAIN_TT', 2: 'SM_TT', 3: 'CAR_TT'},
    'TC': {1: 'TRAIN_TC', 2: 'SM_TC', 3: 'CAR_TC'},
    'HE': {1: 'TRAIN_HE', 2: 'SM_HE'},
    'SEAT': {2: 'SM_SEATS'}
}

availability_variables = {1: 'TRAIN_AV', 2: 'SM_AV', 3: 'CAR_AV'}

custom_alt_id = "mode_id"
obs_id_column = "custom_id"
df_wide[obs_id_column] = np.arange(df_wide.shape[0], dtype=int) + 1
choice_column = "CHOICE"


df_long = pl.convert_wide_to_long(
    df_wide,
    ind_variables,
    alt_varying_variables,
    availability_variables,
    obs_id_column,
    choice_column,
    new_alt_id_name=custom_alt_id
)

df_long["TTxAGE_2"] = df_long["TT"] * df_long["AGE_2"]
df_long["TTxAGE_3"] = df_long["TT"] * df_long["AGE_3"]
df_long["TTxAGE_4"] = df_long["TT"] * df_long["AGE_4"]
df_long["TTxAGE_5"] = df_long["TT"] * df_long["AGE_5"]
df_long["TTxINCOME_2"] = df_long["TT"] * df_long["INCOME_2"]
df_long["TTxINCOME_3"] = df_long["TT"] * df_long["INCOME_3"]
df_long["TTxPURPOSE_2"] = df_long["TT"] * df_long["PURPOSE_2"]
df_long["TTxPURPOSE_3"] = df_long["TT"] * df_long["PURPOSE_3"]
df_long["TTxPURPOSE_4"] = df_long["TT"] * df_long["PURPOSE_4"]



SS_fixed = GroupShuffleSplit(n_splits=2, train_size=0.7, random_state=1234)
train_idx_fixed, rem_idx_fixed = next(SS_fixed.split(df_wide, groups=df_wide['custom_id']))
df_wide_train_fixed = df_wide.iloc[train_idx_fixed]
df_wide_rem_fixed = df_wide.iloc[rem_idx_fixed]

SS2_fixed = GroupShuffleSplit(n_splits=2, train_size=0.5, random_state=1234)
test_idx_fixed, valid_idx_fixed = next(SS2_fixed.split(df_wide_rem_fixed, groups=df_wide_rem_fixed['custom_id']))
df_wide_test_fixed = df_wide_rem_fixed.iloc[test_idx_fixed]
df_wide_valid = df_wide_rem_fixed.iloc[valid_idx_fixed]  # This is our fixed validation set

df_wide_pool = df_wide[~df_wide.custom_id.isin(df_wide_valid.custom_id)]

######################################################################################################
#                                  SYNTHETIC DATA GENERATION
#        
######################################################################################################
np.random.seed(42)

# Means for the parameters (original "current" betas)
alpha_train_mean, alpha_sm_mean, alpha_car_mean = 0.3, 0.5, 0.1
beta_tt_train_mean, beta_tc_train_mean, beta_he_train_mean = -1.2, -1.8, -0.4
beta_tt_sm_mean, beta_tc_sm_mean, beta_he_sm_mean, beta_seat_sm_mean = -1.0, -2.0, -1.2, 0.2
beta_tt_car_mean, beta_tc_car_mean = -1.4, -1.9

# Standard deviations for random draws
std_alpha = 0.01
std_beta_tt = 0.01
std_beta_tc = 0.01
std_beta_he = 0.01
std_beta_seat = 0.01

unique_ids = df_wide["ID"].unique()
n_ids = len(unique_ids)
rng = np.random.default_rng(42)

alpha_train_i = rng.normal(alpha_train_mean, std_alpha, size=n_ids)
alpha_sm_i = rng.normal(alpha_sm_mean, std_alpha, size=n_ids)
alpha_car_i = rng.normal(alpha_car_mean, std_alpha, size=n_ids)

# === behaviourally-consistent β draws (TT, TC, HE) ==========
def draw_neg_lognormal(target_mean, sigma, size):
    """
    Draw strictly NEGATIVE log-normal variates whose mean is ≈ target_mean
    (target_mean must itself be negative). sigma is the shape parameter.
    """
    mu = np.log(abs(target_mean)) - 0.5 * sigma**2
    return -np.random.lognormal(mean=mu, sigma=sigma, size=size)

std_beta_ln = 0.3   # common shape parameter for all log-normal coefficients

# ----- Train alternative ---------------------------------------------------
beta_tt_train_i = draw_neg_lognormal(beta_tt_train_mean, std_beta_ln, n_ids)
beta_tc_train_i = draw_neg_lognormal(beta_tc_train_mean, std_beta_ln, n_ids)
beta_he_train_i = draw_neg_lognormal(beta_he_train_mean, std_beta_ln, n_ids)

# ----- Swissmetro alternative ---------------------------------------------
beta_tt_sm_i   = draw_neg_lognormal(beta_tt_sm_mean  , std_beta_ln, n_ids)
beta_tc_sm_i   = draw_neg_lognormal(beta_tc_sm_mean  , std_beta_ln, n_ids)
beta_he_sm_i   = draw_neg_lognormal(beta_he_sm_mean  , std_beta_ln, n_ids)

# ----- Car alternative -----------------------------------------------------
beta_tt_car_i  = draw_neg_lognormal(beta_tt_car_mean , std_beta_ln, n_ids)
beta_tc_car_i  = draw_neg_lognormal(beta_tc_car_mean , std_beta_ln, n_ids)


beta_seat_sm_i = rng.normal(beta_seat_sm_mean, std_beta_seat, size=n_ids)




df_betas = pd.DataFrame({
    "ID": unique_ids,
    "alpha_train": alpha_train_i,
    "alpha_sm": alpha_sm_i,
    "alpha_car": alpha_car_i,
    "beta_tt_train": beta_tt_train_i,
    "beta_tc_train": beta_tc_train_i,
    "beta_he_train": beta_he_train_i,
    "beta_tt_sm": beta_tt_sm_i,
    "beta_tc_sm": beta_tc_sm_i,
    "beta_he_sm": beta_he_sm_i,
    "beta_seat_sm": beta_seat_sm_i,
    "beta_tt_car": beta_tt_car_i,
    "beta_tc_car": beta_tc_car_i
})

# Merge these random coefficients onto df_wide
df_wide = pd.merge(df_wide, df_betas, on="ID", how="left")


def util_train(row):
    # Revised utility with negative interaction term to ensure monotonic decreasing cost effect
    return (row["alpha_train"]
            + row["beta_tt_train"] * np.log(1 + row["TRAIN_TT"])
            + row["beta_tc_train"] * row["TRAIN_TC"]
            + row["beta_he_train"] * row["TRAIN_HE"] * row["INCOMES"]
            - 0.05 * row["AGES"] * row["TRAIN_TC"])


def util_swiss(row):
    return (row["alpha_sm"]
            + row["beta_tt_sm"] * (row["SM_TT"] ** 1.5)
            + row["beta_tc_sm"] * np.log(1 + row["SM_TC"])
            + row["beta_he_sm"] * np.sqrt(1 + row["SM_HE"])
            + row["beta_seat_sm"] * row.get("SM_SEATS", 0.0)
            - 0.1 * row["AGES"] * row["SM_HE"] * row["INCOMES"])


def util_car(row):
    return (row["alpha_car"]
            + row["beta_tt_car"] * (row["CAR_TT"] ** 2)
            + row["beta_tc_car"] * row["CAR_TC"]
            - 0.1 * row["PURPOSES"] * np.log(1 + row["CAR_TC"] * row["CAR_TT"]))


df_wide["util_train"] = df_wide.apply(util_train, axis=1)
df_wide["util_sm"] = df_wide.apply(util_swiss, axis=1)
df_wide["util_car"] = df_wide.apply(util_car, axis=1)
# --- DEBUG: peek at car utilities ---
print("util_car: shape =", df_wide["util_car"].shape)
print("util_car: first 5 values =", df_wide["util_car"].head().values)


U_mat = df_wide[["util_train", "util_sm", "util_car"]].values
expU = np.exp(U_mat)
den = expU.sum(axis=1, keepdims=True)
probabilities = expU / den
print("probabilities shape:", probabilities.shape)
print("first 5 choice‐prob rows:", probabilities[:5])
df_wide["CHOICE"] = np.array([np.random.choice([1, 2, 3], p=p) for p in probabilities])

# Compute individual-level "true" VOT & VOH from the derivatives of the utilities

df_wide["VOT_train_row"] = (df_wide["beta_tt_train"] / (1 + df_wide["TRAIN_TT"])) / (
            df_wide["beta_tc_train"] - 0.05 * df_wide["AGES"])

df_wide["VOT_sm_row"] = (1.5 * df_wide["beta_tt_sm"] * (df_wide["SM_TT"] ** 0.5)) / (
            df_wide["beta_tc_sm"] / (1 + df_wide["SM_TC"]))

df_wide["VOT_car_row"] = (2 * df_wide["beta_tt_car"] * df_wide["CAR_TT"] - 0.1 * df_wide["PURPOSES"] * (
            df_wide["CAR_TC"] / (1 + df_wide["CAR_TC"] * df_wide["CAR_TT"]))) / (
                                     df_wide["beta_tc_car"] - 0.1 * df_wide["PURPOSES"] * (
                                         df_wide["CAR_TT"] / (1 + df_wide["CAR_TC"] * df_wide["CAR_TT"])))

df_wide["VOH_sm_row"] = ((df_wide["beta_he_sm"] / (2 * np.sqrt(1 + df_wide["SM_HE"]))) - 0.1 * df_wide["AGES"] *
                         df_wide["INCOMES"]) / (df_wide["beta_tc_sm"] / (1 + df_wide["SM_TC"]))

VOT_train_ground = _mis_spec(df_wide["VOT_train_row"].mean())
VOT_sm_ground    = _mis_spec(df_wide["VOT_sm_row"].mean())
VOT_car_ground   = _mis_spec(df_wide["VOT_car_row"].mean())

VOH_sm_ground = df_wide["VOH_sm_row"].mean()

print("\n---- Mean 'True' VOT & VOH from derivative-based calculations ----")
print("VOT_train_ground =", VOT_train_ground)
print("VOT_sm_ground    =", VOT_sm_ground)
print("VOT_car_ground   =", VOT_car_ground)
print("VOH_sm_ground    =", VOH_sm_ground)


directory = 'ASU_ResBINN_Models/'
if not os.path.exists(directory):
    os.makedirs(directory)

directory1 = 'ASU_ResBINN_Results/'
if not os.path.exists(directory1):
    os.makedirs(directory1)


######################################################################################################
#                      DOMAIN KNOWLEDGE LOSS (PINNSyntheticLoss)
#                    
######################################################################################################
class PINNSyntheticLoss(tf.keras.losses.Loss):
    def __init__(self, model, lambda_reg=0.1, lambda_neg=0.1, k=4, k_prime=10, re_norm_epoch=2, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.lambda_reg = tf.constant(lambda_reg, dtype=tf.float32)
        self.lambda_neg = tf.constant(lambda_neg, dtype=tf.float32)
        self.k = k
        self.k_prime = k_prime
        self.initial_data_loss = tf.Variable(0.0, trainable=False)
        self.initial_physics_loss = tf.Variable(0.0, trainable=False)
        self.initial_neg_sign_loss = tf.Variable(0.0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.re_norm_epoch = re_norm_epoch
        self.adapt_interval = 2
        self.step_counter = tf.Variable(0, trainable=False)
        self.data_baseline = tf.Variable(1.0, trainable=False)
        self.physics_baseline = tf.Variable(1.0, trainable=False)
        self.neg_baseline = tf.Variable(1.0, trainable=False)

    def adaptive_scaling(self, current_loss, baseline):
        return tf.clip_by_value(current_loss / (baseline + 1e-12), 0.1, 10.0)

    def compute_corrected_ratio(self, WTP_X, WTP_Y, k_prime):
        ratio_XY = WTP_X / (WTP_Y + 1e-6)
        sorted_indices = tf.argsort(ratio_XY)
        sorted_WTP_X = tf.gather(WTP_X, sorted_indices)
        sorted_WTP_Y = tf.gather(WTP_Y, sorted_indices)
        n = tf.shape(sorted_WTP_X)[0]

        quantiles = tf.cast(tf.linspace(0.0, 1.0, k_prime + 1), tf.float32)
        indices = tf.cast(tf.cast(n - 1, tf.float32) * quantiles, tf.int32)

        def body(i, group_ratios):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            group_ratio = tf.cond(
                start_idx >= end_idx,
                lambda: tf.constant(0.0, dtype=tf.float32),
                lambda: tf.reduce_mean(sorted_WTP_X[start_idx:end_idx]) /
                        (tf.reduce_mean(sorted_WTP_Y[start_idx:end_idx]) + 1e-6)
            )
            return i + 1, tf.concat([group_ratios, [group_ratio]], axis=0)

        _, group_ratios = tf.while_loop(
            cond=lambda i, _: i < k_prime,
            body=body,
            loop_vars=(0, tf.zeros([0], dtype=tf.float32)),
            shape_invariants=(tf.TensorShape([]), tf.TensorShape([None]))
        )

        valid_mask = tf.not_equal(group_ratios, 0.0)
        valid_ratios = tf.boolean_mask(group_ratios, valid_mask)
        return tf.cond(
            tf.size(valid_ratios) > 0,
            lambda: tf.reduce_mean(valid_ratios),
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        self.epoch.assign_add(1)

        # Data-fitting loss
        data_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))

        
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_train_tensor)
            model_outputs = self.model(x_train_tensor, training=False)
            prob, util1, util2, util3 = model_outputs

            grad_train = tape.gradient(util1, x_train_tensor)
            grad_sm = tape.gradient(util2, x_train_tensor)
            grad_car = tape.gradient(util3, x_train_tensor)

            # Gradient Surgery: apply PCGrad to reduce interference between tasks
            def pc_grad(grads):
                new_grads = []
                for i in range(len(grads)):
                    g_i = grads[i]
                    for j in range(len(grads)):
                        if i == j:
                            continue
                        dot_ij = tf.reduce_sum(g_i * grads[j], axis=1, keepdims=True)
                        norm_j_sq = tf.reduce_sum(tf.square(grads[j]), axis=1, keepdims=True) + 1e-6
                        projection = (dot_ij / norm_j_sq) * grads[j]
                        condition = tf.cast(dot_ij < 0, tf.float32)
                        g_i = g_i - condition * projection
                    new_grads.append(g_i)
                return new_grads

            grad_train, grad_sm, grad_car = pc_grad([grad_train, grad_sm, grad_car])

            # x_train columns: 0:TRAIN_TT,1:TRAIN_TC,2:TRAIN_HE,3:SM_TT,4:SM_TC,5:SM_HE,6:SM_SEATS,7:CAR_TT,8:CAR_TC,9:GA,10:AGES,11:INCOMES,12:PURPOSES
            marginal_utility_TT_nn_train = grad_train[:, 0]
            marginal_utility_TC_nn_train = grad_train[:, 1]

            marginal_utility_SM_TT_nn = grad_sm[:, 3]
            marginal_utility_SM_TC_nn = grad_sm[:, 4]

            marginal_utility_CAR_TT_nn = grad_car[:, 7]
            marginal_utility_CAR_TC_nn = grad_car[:, 8]
        # Compute WTP = derivative wrt time / derivative wrt cost
        WTP_nn_train = marginal_utility_TT_nn_train / (marginal_utility_TC_nn_train + 1e-6)
        WTP_nn_sm = marginal_utility_SM_TT_nn / (marginal_utility_SM_TC_nn + 1e-6)
        WTP_nn_car = marginal_utility_CAR_TT_nn / (marginal_utility_CAR_TC_nn + 1e-6)

        # Domain-based constraints on expected VOT
        E_X = tf.constant(VOT_train_ground, dtype=tf.float32)  # single scalar = mean Train VOT
        E_Y = tf.constant(VOT_sm_ground, dtype=tf.float32)  # single scalar = mean SM VOT
        E_Z = tf.constant(VOT_car_ground, dtype=tf.float32)  # single scalar = mean Car VOT


        err_train = tf.abs(tf.reduce_mean(WTP_nn_train) - E_X)
        err_sm = tf.abs(tf.reduce_mean(WTP_nn_sm) - E_Y)
        err_car = tf.abs(tf.reduce_mean(WTP_nn_car) - E_Z)
        phys = tf.stack([err_train, err_sm, err_car], axis=0)  # shape=(3,)

       
        eps = 1e-3
        p = 1.0  
        inv = tf.pow(1.0 / (phys + eps), p)  

       
        weights = inv / tf.reduce_sum(inv) 

   
        physics_loss = tf.reduce_sum(weights * phys)




        def negative_sign_loss(mu):
            return tf.reduce_mean(tf.maximum(0.0, mu))

        neg_sign_loss = (
                                negative_sign_loss(marginal_utility_TT_nn_train) +
                                negative_sign_loss(marginal_utility_TC_nn_train) +
                                negative_sign_loss(marginal_utility_SM_TT_nn) +
                                negative_sign_loss(marginal_utility_SM_TC_nn) +
                                negative_sign_loss(marginal_utility_CAR_TT_nn) +
                                negative_sign_loss(marginal_utility_CAR_TC_nn)
                        ) / 6.0

  
        if self.step_counter % self.adapt_interval == 0:
            self.data_baseline.assign(data_loss)
            self.physics_baseline.assign(physics_loss)
            self.neg_baseline.assign(neg_sign_loss)

        data_scale = self.adaptive_scaling(data_loss, self.data_baseline)
        physics_scale = self.adaptive_scaling(physics_loss, self.physics_baseline)
        neg_scale = self.adaptive_scaling(neg_sign_loss, self.neg_baseline)

        self.step_counter.assign_add(1)
        total_loss = (
                data_scale * data_loss +
                self.lambda_reg * physics_scale * physics_loss +
                self.lambda_neg * neg_scale * neg_sign_loss
        )
        return total_loss


######################################################################################################
#                     ASU_ResBINN MODEL BUILD FUNCTION
######################################################################################################
def build_ASU_ResBINN(num_layers, num_neurons, drop_rate, learning_rate,
                      lambda_reg=0.1, lambda_neg=0.1, k=4,
                      k_prime=10, re_norm_epoch=2,num_heads=4):
    input_layer = Input(shape=(x_train.shape[1],), name="main_input", dtype=tf.float64)
    # columns: 0:TRAIN_TT,1:TRAIN_TC,2:TRAIN_HE,3:SM_TT,4:SM_TC,5:SM_HE,6:SM_SEATS,7:CAR_TT,8:CAR_TC,9:GA,10:AGES,11:INCOMES,12:PURPOSES
    train_inputs = tf.keras.layers.Lambda(lambda x: x[:, :3], name="train_inputs")(input_layer)
    sm_inputs = tf.keras.layers.Lambda(lambda x: x[:, 3:7], name="sm_inputs")(input_layer)
    car_inputs = tf.keras.layers.Lambda(lambda x: x[:, 7:9], name="car_inputs")(input_layer)
    socio_demographic_inputs = tf.keras.layers.Lambda(lambda x: x[:, 9:], name="socio_inputs")(input_layer)

    def alternative_subnetwork(
            alt_inputs,
            socio_inputs,
            num_layers,
            num_neurons,
            drop_rate,
            name_prefix):
        # --- 1. embed alternative & person ---
        h_attr = Dense(num_neurons, activation='relu',
                       name=f"{name_prefix}_attr_embed")(alt_inputs)  # [B, d]
        h_pers = Dense(num_neurons, activation='relu',
                       name=f"{name_prefix}_pers_embed")(socio_inputs)  # [B, d]

        # --- 2. FiLM gate: γ, β depend on the person vector ---------------
        gamma = Dense(num_neurons, activation='sigmoid',
                      name=f"{name_prefix}_gamma")(h_pers)  # [B, d]
        beta = Dense(num_neurons, activation='tanh',
                     name=f"{name_prefix}_beta")(h_pers)  # [B, d]
        h = gamma * h_attr + beta  # FiLM

        h = LayerNormalization(name=f"{name_prefix}_norm")(h)

        # --- 3. residual MLP  ---------------------------------
        for i in range(num_layers):
            skip = h
            h = Dense(num_neurons, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                      kernel_constraint=tf.keras.constraints.max_norm(3),
                      name=f"{name_prefix}_dense_{i + 1}")(h)
            h = Dropout(drop_rate, name=f"{name_prefix}_drop_{i + 1}")(h)
            h = Add(name=f"{name_prefix}_resadd_{i + 1}")([skip, h])

        out_utility = Dense(1, activation='linear',
                            name=f"{name_prefix}_utility")(h)
        return out_utility

    util_ALT1 = alternative_subnetwork(train_inputs, socio_demographic_inputs,
                                       num_layers, num_neurons, drop_rate, "train")
    util_ALT2 = alternative_subnetwork(sm_inputs, socio_demographic_inputs,
                                       num_layers, num_neurons, drop_rate, "sm")
    util_ALT3 = alternative_subnetwork(car_inputs, socio_demographic_inputs,
                                       num_layers, num_neurons, drop_rate, "car")

    concatenated_utils = Concatenate(name='concatenate_utils')([util_ALT1, util_ALT2, util_ALT3])
    out_prob = tf.keras.layers.Softmax(name='out_prob')(concatenated_utils)

    model = Model(inputs=input_layer, outputs=[out_prob, util_ALT1, util_ALT2, util_ALT3])
    loss_instance = PINNSyntheticLoss(model,
                                      lambda_reg=lambda_reg * (1.0 + 0.5 * num_heads),
                                      lambda_neg=lambda_neg,
                                      k=k,
                                      k_prime=k_prime,
                                      re_norm_epoch=re_norm_epoch)
    optimizer = Adam(learning_rate=learning_rate, clipnorm=0.5)
    model.compile(
        optimizer=optimizer,
        loss=[loss_instance, None, None, None],
        metrics=['accuracy', None, None, None]
    )
    return model
##############################################################################
# ----------------  ANALYTIC VOT / VOH  (gradient method) -------------------
##############################################################################
def analytic_vot_voh(model, X):
    """Return VOT_train, VOT_sm, VOT_car, VOH_sm (per row)."""
    Xtf = tf.convert_to_tensor(X, tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(Xtf)
        _, u_tr, u_sm, u_car = model(Xtf, training=False)

    dU_tr_TT  = tape.gradient(u_tr,  Xtf)[:, 0]
    dU_tr_TC  = tape.gradient(u_tr,  Xtf)[:, 1]
    dU_sm_TT  = tape.gradient(u_sm,  Xtf)[:, 3]
    dU_sm_TC  = tape.gradient(u_sm,  Xtf)[:, 4]
    dU_car_TT = tape.gradient(u_car, Xtf)[:, 7]
    dU_car_TC = tape.gradient(u_car, Xtf)[:, 8]
    dU_sm_HE  = tape.gradient(u_sm,  Xtf)[:, 5]

    EPS = 1e-6
    vot_train = (dU_tr_TT  / (dU_tr_TC + EPS)).numpy()
    vot_sm    = (dU_sm_TT  / (dU_sm_TC + EPS)).numpy()
    vot_car   = (dU_car_TT / (dU_car_TC + EPS)).numpy()
    voh_sm    = (dU_sm_HE  / (dU_sm_TC + EPS)).numpy()
    return vot_train, vot_sm, vot_car, voh_sm


##############################################################################
# -----------------  PD / ICE PLOT HELPER   ----------------
##############################################################################
def plot_with_true_overlay(model_est, true_est, feature_name, alt_target,
                           alt_label, savename):
    fig, ax = plt.subplots(figsize=(7, 5))
    disp_model = PartialDependenceDisplay.from_estimator(
        model_est, x_train, [feature_name], kind="both",
        ice_lines_kw=dict(color="gray", alpha=.3, label="_nolegend_"),
        pd_line_kw=dict(color="blue", lw=2, linestyle="-.", label="ASU-ResBINN (Avg)"),
        target=alt_target, subsample=.2, grid_resolution=20, centered=True,
        random_state=1, response_method="decision_function",
        percentiles=(.01, .99), ax=ax)

    PartialDependenceDisplay.from_estimator(
        true_est, x_train, [feature_name], kind="average",
        pd_line_kw=dict(color="black", lw=2, label="True (Avg)"),
        target=alt_target, subsample=.2, grid_resolution=20, centered=True,
        random_state=1, response_method="decision_function",
        percentiles=(.01, .99), ax=disp_model.axes_)

    ax.set_ylabel(f"Utility of {alt_label}")
    ax.set_title(f"{alt_label} Utility vs {feature_name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(custom_plot_dir, savename))
    plt.close(fig)

# ============================================================================#
#  TRUE-VALUE TABLES 
# ============================================================================#
df_true_vot = (
    df_wide[['AGES', 'INCOMES', 'PURPOSES',
             'VOT_train_row', 'VOT_sm_row', 'VOT_car_row']]
    .groupby(['AGES', 'INCOMES', 'PURPOSES'])
    .median()
    .reset_index()
)

df_true_voh = (
    df_wide[['AGES', 'INCOMES', 'PURPOSES', 'VOH_sm_row']]
    .groupby(['AGES', 'INCOMES', 'PURPOSES'])
    .median()
    .reset_index()
)

##############################################################################
# ----------------------  MONTE-CARLO EXPERIMENT LOOP  -----------------------
##############################################################################
train_acc_list, test_acc_list   = [], []
train_brier_list, test_brier_list = [], []
train_ll_list, test_ll_list     = [], []

vot_tr_q_list, vot_sm_q_list, vot_car_q_list, voh_sm_q_list = [], [], [], []

rmse_train_list, mape_train_list = [], []
rmse_sm_list,    mape_sm_list    = [], []
rmse_car_list,   mape_car_list   = [], []
rmse_voh_list,   mape_voh_list   = [], []

results_dict = {}
for i, σ in enumerate(SIGMA_VALUES):
    WTP_MIS_SPEC = σ
    global _rng_wtp
    _rng_wtp = np.random.default_rng(12345 + i)

    # --- RECOMPUTE mis-specified priors *for this experiment* --------------
    VOT_train_ground = _mis_spec(df_wide["VOT_train_row"].mean())
    VOT_sm_ground    = _mis_spec(df_wide["VOT_sm_row"].mean())
    VOT_car_ground   = _mis_spec(df_wide["VOT_car_row"].mean())

    print(f"\n===== BEGIN MONTE CARLO EXPERIMENT {i} with WTP MIS-SPEC σ = {WTP_MIS_SPEC:.2f} =====")
    print("  → Mis-specified VOT priors used:")
    print(f"    VOT_train_ground: {VOT_train_ground:.3f}, VOT_sm_ground: {VOT_sm_ground:.3f}, VOT_car_ground: {VOT_car_ground:.3f}")


    # --------------------------------------------------------------------

    # Re-draw a train/test from df_wide_pool each time (keeping df_wide_valid fixed).
    SS_exp = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=1234 + i)
    train_idx_this, test_idx_this = next(SS_exp.split(df_wide_pool, groups=df_wide_pool['custom_id']))
    df_wide_train = df_wide_pool.iloc[train_idx_this]
    df_wide_test = df_wide_pool.iloc[test_idx_this]

    global x_train, x_test
    x_train = df_wide_train[[
        'TRAIN_TT', 'TRAIN_TC', 'TRAIN_HE',
        'SM_TT', 'SM_TC', 'SM_HE', 'SM_SEATS',
        'CAR_TT', 'CAR_TC',
        'GA', 'AGES', 'INCOMES', 'PURPOSES'
    ]]
    y_train_raw = np.array(pd.get_dummies(df_wide_train['CHOICE']))

    x_test = df_wide_test[[
        'TRAIN_TT', 'TRAIN_TC', 'TRAIN_HE',
        'SM_TT', 'SM_TC', 'SM_HE', 'SM_SEATS',
        'CAR_TT', 'CAR_TC',
        'GA', 'AGES', 'INCOMES', 'PURPOSES'
    ]]
    y_test_raw = np.array(pd.get_dummies(df_wide_test['CHOICE']))

    x_valid = df_wide_valid[[
        'TRAIN_TT', 'TRAIN_TC', 'TRAIN_HE',
        'SM_TT', 'SM_TC', 'SM_HE', 'SM_SEATS',
        'CAR_TT', 'CAR_TC',
        'GA', 'AGES', 'INCOMES', 'PURPOSES'
    ]]
    y_valid_raw = np.array(pd.get_dummies(df_wide_valid['CHOICE']))

    # Build the ASU_ResBINN
    k = 128
    k_prime = 4
    re_norm_epoch = 2
    momentum = 0.9
    num_layers = 3
    num_neurons = 300
    drop_rate = 0.15
    learning_rate = 0.0002
    lambda_reg = 10
    lambda_monotonicity = 10

    num_heads = 4  ## not necessary ---for old implementation


    model = build_ASU_ResBINN(num_layers, num_neurons, drop_rate, learning_rate,
                                    lambda_reg=lambda_reg, lambda_neg=lambda_monotonicity,
                                    k=k, k_prime=k_prime, re_norm_epoch=re_norm_epoch,num_heads=num_heads)

    y_train_tensor = tf.convert_to_tensor(y_train_raw, dtype=tf.float32)
    y_valid_tensor = tf.convert_to_tensor(y_valid_raw, dtype=tf.float32)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True,
                                          verbose=1, patience=30)


    history = model.fit(
        x=x_train,
        y=y_train_tensor,
        shuffle=True,
        epochs=200,
        batch_size=128,
        validation_data=[x_valid, y_valid_tensor],
        callbacks=[es],
        verbose=1
    )

    best_val_loss = min(history.history['val_loss'])
    if best_val_loss >= 1.1:
        print(f"Experiment {i} (σ={σ}): No convergence (best val_loss={best_val_loss:.3f}), skipping.")
        continue
    # ---------------- metrics ---------------------------------------------
    train_p = model.predict(x_train, 0)[0]
    test_p = model.predict(x_test, 0)[0]

    train_acc = accuracy_score(y_train_raw.argmax(1), train_p.argmax(1))
    test_acc = accuracy_score(y_test_raw.argmax(1), test_p.argmax(1))
    train_brier = brier_multi(y_train_raw, train_p);
    test_brier = brier_multi(y_test_raw, test_p)
    train_ll = F.nll_loss(torch.log(torch.tensor(train_p)),
                          torch.tensor(y_train_raw.argmax(1))).item()
    test_ll = F.nll_loss(torch.log(torch.tensor(test_p)),
                         torch.tensor(y_test_raw.argmax(1))).item()

    train_acc_list.append(train_acc);
    test_acc_list.append(test_acc)
    train_brier_list.append(train_brier);
    test_brier_list.append(test_brier)
    train_ll_list.append(train_ll);
    test_ll_list.append(test_ll)

    print(f"Accuracy  Train/Test : {train_acc:.3f} / {test_acc:.3f}")

    # ---------------- analytic VOT / VOH ----------------------------------
    vot_tr, vot_sm, vot_car, voh_sm = analytic_vot_voh(model, x_train)

    vot_tr_q_list.append(np.quantile(vot_tr, (0.5, .01, .25, .75, .99)).round(3))
    vot_sm_q_list.append(np.quantile(vot_sm, (0.5, .01, .25, .75, .99)).round(3))
    vot_car_q_list.append(np.quantile(vot_car, (0.5, .01, .25, .75, .99)).round(3))
    voh_sm_q_list.append(np.quantile(voh_sm, (0.5, .01, .25, .75, .99)).round(3))

    print("VOT Train Qs:", vot_tr_q_list[-1])
    print("VOT SM    Qs:", vot_sm_q_list[-1])
    print("VOT Car   Qs:", vot_car_q_list[-1])

    # ---------------- PD / ICE sanity plots -------------------------------
    custom_plot_dir = 'ResBINN_Montecarlo'
    if not os.path.exists(custom_plot_dir): os.makedirs(custom_plot_dir)


    class Estimator(BaseEstimator, ClassifierMixin):
        def fit(self, X, y): self.classes_ = unique_labels(y); return self

        def predict_proba(self, X): return model.predict(X, 0)[0]

        def decision_function(self, X):
            p, u1, u2, u3 = model.predict(X, 0);
            return np.hstack([u1, u2, u3])


    class TrueEst(BaseEstimator, ClassifierMixin):
        def fit(self, X, y): self.classes_ = np.array([0, 1, 2]); return self

        def decision_function(self, X):
            X = np.asarray(X)
            u_train = (alpha_train_mean
                       + beta_tt_train_mean * np.log1p(X[:, 0])
                       + beta_tc_train_mean * X[:, 1]
                       + beta_he_train_mean * X[:, 2] * X[:, 11]
                       - 0.05 * X[:, 10] * X[:, 1])
            u_sm = (alpha_sm_mean + beta_tt_sm_mean * X[:, 3] ** 1.5
                    + beta_tc_sm_mean * np.log1p(X[:, 4])
                    + beta_he_sm_mean * np.sqrt(1 + X[:, 5])
                    + beta_seat_sm_mean * X[:, 6]
                    - 0.1 * X[:, 10] * X[:, 5] * X[:, 11])
            u_car = (alpha_car_mean + beta_tt_car_mean * X[:, 7] ** 2
                     + beta_tc_car_mean * X[:, 8]
                     - 0.1 * X[:, 12] * np.log1p(X[:, 8] * X[:, 7]))
            return np.c_[u_train, u_sm, u_car]

        def predict_proba(self, X):
            util = self.decision_function(X);
            e = np.exp(util);
            return e / e.sum(1, keepdims=True)


    # ------------------------------------------------------------------
    # ---- PD / ICE  --------------
    # ------------------------------------------------------------------
    est = Estimator().fit(x_train, y_train_raw)  
    tru = TrueEst().fit(x_train, y_train_raw)

    plot_with_true_overlay(est, tru, "TRAIN_TT", 0, "Train",
                           f"train_TT_{i}.png")
    plot_with_true_overlay(est, tru, "TRAIN_TC", 0, "Train",
                           f"train_TC_{i}.png")
    plot_with_true_overlay(est, tru, "SM_TT", 1, "Swissmetro",
                           f"sm_TT_{i}.png")
    plot_with_true_overlay(est, tru, "SM_TC", 1, "Swissmetro",
                           f"sm_TC_{i}.png")
    plot_with_true_overlay(est, tru, "CAR_TT", 2, "Car",
                           f"car_TT_{i}.png")
    plot_with_true_overlay(est, tru, "CAR_TC", 2, "Car",
                           f"car_TC_{i}.png")

    # ---------------- aggregated group-level RMSE / MAPE ------------------
    df_pred_vot = x_train[['AGES', 'INCOMES', 'PURPOSES']].copy()
    df_pred_vot['VOT_train'], df_pred_vot['VOT_sm'], df_pred_vot['VOT_car'] = vot_tr, vot_sm, vot_car
    grp_pred_vot = df_pred_vot.groupby(['AGES', 'INCOMES', 'PURPOSES']).median().reset_index()
    df_true_vot_merge = pd.merge(df_true_vot, grp_pred_vot,
                                 on=['AGES', 'INCOMES', 'PURPOSES'])

    rmse_train = np.sqrt(((df_true_vot_merge['VOT_train_row'] - df_true_vot_merge['VOT_train']) ** 2).mean())
    mape_train = np.abs((df_true_vot_merge['VOT_train_row'] - df_true_vot_merge['VOT_train'])
                        / df_true_vot_merge['VOT_train_row']).mean() * 100
    rmse_sm = np.sqrt(((df_true_vot_merge['VOT_sm_row'] - df_true_vot_merge['VOT_sm']) ** 2).mean())
    mape_sm = np.abs((df_true_vot_merge['VOT_sm_row'] - df_true_vot_merge['VOT_sm'])
                     / df_true_vot_merge['VOT_sm_row']).mean() * 100
    rmse_car = np.sqrt(((df_true_vot_merge['VOT_car_row'] - df_true_vot_merge['VOT_car']) ** 2).mean())
    mape_car = np.abs((df_true_vot_merge['VOT_car_row'] - df_true_vot_merge['VOT_car'])
                      / df_true_vot_merge['VOT_car_row']).mean() * 100

    df_pred_voh = x_train[['AGES', 'INCOMES', 'PURPOSES']].copy()
    df_pred_voh['VOH_sm'] = voh_sm
    grp_pred_voh = df_pred_voh.groupby(['AGES', 'INCOMES', 'PURPOSES']).median().reset_index()
    df_true_voh_merge = pd.merge(df_true_voh, grp_pred_voh,
                                 on=['AGES', 'INCOMES', 'PURPOSES'])
    rmse_voh = np.sqrt(((df_true_voh_merge['VOH_sm_row'] - df_true_voh_merge['VOH_sm']) ** 2).mean())
    mape_voh = np.abs((df_true_voh_merge['VOH_sm_row'] - df_true_voh_merge['VOH_sm'])
                      / df_true_voh_merge['VOH_sm_row']).mean() * 100

    rmse_train_list.append(rmse_train);
    mape_train_list.append(mape_train)
    rmse_sm_list.append(rmse_sm);
    mape_sm_list.append(mape_sm)
    rmse_car_list.append(rmse_car);
    mape_car_list.append(mape_car)
    rmse_voh_list.append(rmse_voh);
    mape_voh_list.append(mape_voh)

    results_dict[σ] = dict(train_acc=train_acc, test_acc=test_acc,
                           rmse_train=rmse_train, mape_train=mape_train,
                           rmse_sm=rmse_sm, mape_sm=mape_sm,
                           rmse_car=rmse_car, mape_car=mape_car)

    print(f"RMSE VOT Train/SM/Car : {rmse_train:.3f}/{rmse_sm:.3f}/{rmse_car:.3f}")
    print(f"MAPE VOT Train/SM/Car : {mape_train:.2f}%/{mape_sm:.2f}%/{mape_car:.2f}%")
    print(f"RMSE VOH SM           : {rmse_voh:.3f}  |  MAPE {mape_voh:.2f}%")
    print("------------------------------------------------------------------")

# =====================  FINAL AGGREGATE PRINTS  =============================
print("\n===== ACCURACY / NLL / BRIER (μ ± σ across runs) =====")
print(f"Train acc : {np.mean(train_acc_list):.3f} ± {np.std(train_acc_list):.3f}")
print(f"Test  acc : {np.mean(test_acc_list):.3f} ± {np.std(test_acc_list):.3f}")
print(f"Train NLL : {np.mean(train_ll_list):.3f} ± {np.std(train_ll_list):.3f}")
print(f"Test  NLL : {np.mean(test_ll_list):.3f} ± {np.std(test_ll_list):.3f}")
print(f"Train Bri : {np.mean(train_brier_list):.3f} ± {np.std(train_brier_list):.3f}")
print(f"Test  Bri : {np.mean(test_brier_list):.3f} ± {np.std(test_brier_list):.3f}")


def prn_q(name, arr_list):
    arr = np.array(arr_list);
    mean, sd = arr.mean(0).round(3), arr.std(0).round(3)
    qs = ["Median", "P1", "Q1", "Q3", "P99"]
    print(f"\n{name} quantiles:")
    for q, m, s in zip(qs, mean, sd): print(f"{q:<7}: {m} ± {s}")


prn_q("VOT Train", vot_tr_q_list)
prn_q("VOT SM", vot_sm_q_list)
prn_q("VOT Car", vot_car_q_list)
prn_q("VOH SM", voh_sm_q_list)

print("\n===== AGGREGATE RMSE / MAPE =====")
print(f"RMSE VOT  T/S/C : {np.mean(rmse_train_list):.3f} / "
      f"{np.mean(rmse_sm_list):.3f} / {np.mean(rmse_car_list):.3f}")
print(f"MAPE VOT  T/S/C : {np.mean(mape_train_list):.2f}% / "
      f"{np.mean(mape_sm_list):.2f}% / {np.mean(mape_car_list):.2f}%")
print(f"RMSE VOH SM     : {np.mean(rmse_voh_list):.3f}  |  "
      f"MAPE VOH SM : {np.mean(mape_voh_list):.2f}%")


# =====================  FINAL AGGREGATE PRINTS  =============================


# Compute means and stds
rmse_train_mean, rmse_train_std = np.mean(rmse_train_list), np.std(rmse_train_list)
rmse_sm_mean,    rmse_sm_std    = np.mean(rmse_sm_list),    np.std(rmse_sm_list)
rmse_car_mean,   rmse_car_std   = np.mean(rmse_car_list),   np.std(rmse_car_list)
rmse_voh_mean,   rmse_voh_std   = np.mean(rmse_voh_list),   np.std(rmse_voh_list)

mape_train_mean, mape_train_std = np.mean(mape_train_list), np.std(mape_train_list)
mape_sm_mean,    mape_sm_std    = np.mean(mape_sm_list),    np.std(mape_sm_list)
mape_car_mean,   mape_car_std   = np.mean(mape_car_list),   np.std(mape_car_list)
mape_voh_mean,   mape_voh_std   = np.mean(mape_voh_list),   np.std(mape_voh_list)

print("\n===== AGGREGATE RMSE / MAPE  with std=====")
print(f"RMSE VOT  T/S/C : "
      f"{rmse_train_mean:.3f} ± {rmse_train_std:.3f} / "
      f"{rmse_sm_mean:.3f} ± {rmse_sm_std:.3f} / "
      f"{rmse_car_mean:.3f} ± {rmse_car_std:.3f}")
print(f"MAPE VOT  T/S/C : "
      f"{mape_train_mean:.2f}% ± {mape_train_std:.2f}% / "
      f"{mape_sm_mean:.2f}% ± {mape_sm_std:.2f}% / "
      f"{mape_car_mean:.2f}% ± {mape_car_std:.2f}%")
print(f"RMSE VOH SM     : {rmse_voh_mean:.3f} ± {rmse_voh_std:.3f}  |  "
      f"MAPE VOH SM : {mape_voh_mean:.2f}% ± {mape_voh_std:.2f}%")




print("\n===== SUMMARY PER σ =====")
for σ in SIGMA_VALUES:
    if σ in results_dict:
        r = results_dict[σ]
        print(f"σ={σ:.2f}  accT/accS={r['train_acc']:.3f}/{r['test_acc']:.3f}  "
              f"RMSE T/S/C={r['rmse_train']:.3f}/{r['rmse_sm']:.3f}/{r['rmse_car']:.3f}")
    else:
        print(f"σ={σ:.2f}  – run skipped (no convergence)")
