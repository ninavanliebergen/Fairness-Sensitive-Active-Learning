from pandas import read_csv, to_datetime, get_dummies, concat
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import modAL
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import matplotlib.pyplot as plt
import matplotlib as mpl

import random

from modAL.uncertainty import uncertainty_sampling, classifier_entropy, classifier_margin, classifier_uncertainty, entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.models import Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling

import time
import csv

import fairlearn
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"

import sys
import time

# import all the helpers functions:
from helpers import *

# Function for random sampling
def active_learning_random_profiles(X, y, S, n_start, n_instances, profiles, seed, fairness_metric, n_queries=500, model_mode=LogisticRegression):
    performance_history_profiles = {key: [] for key in np.unique(profiles)}
    fairness_history_profiles = {key: [] for key in np.unique(profiles)}
    f1_scores_history_profiles = {key: [] for key in np.unique(profiles)}

    performance_history = []
    fairness_history = []
    f1_score_history = []

    instances = []

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, S_train, S_test, profiles_train, profiles_test = train_test_split(X, y, S, profiles, test_size=0.3, random_state=seed)

    np.random.seed(seed)
    random_indices = np.random.choice(X_train.shape[0], size=n_start, replace=False)

    X_train_start = X_train[random_indices]
    y_train_start = y_train[random_indices]
    S_train_start = S_train[random_indices]

    X_train_pool = np.delete(X_train, random_indices, axis=0)
    y_train_pool = np.delete(y_train, random_indices, axis=0)
    S_train_pool = np.delete(S_train, random_indices, axis=0)

    learner = ActiveLearner(
        estimator=LogisticRegression(max_iter=10000),
        X_training=X_train_start,
        y_training=y_train_start,
    )

    if model_mode != LogisticRegression:
        learner = ActiveLearner(
            estimator=RandomForestClassifier(),
            X_training=X_train_start,
            y_training=y_train_start
        )

    profile_test_data = profile_data_dict(profiles_test, X_train_pool)
    profile_test_labels = profile_labels_dict(profiles_test, y_train_pool)
    profile_test_S = profile_S_dict(profiles_test, S_train_pool)

    total_instances = n_start

    np.random.seed()
    for _ in range(n_queries):
        y_pred = learner.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        performance_history.append(accuracy)

        f1 = f1_score(y_test, y_pred)
        f1_score_history.append(f1)

        instances.append(total_instances)

        fairness = fairness_metric(y_test, y_pred, sensitive_features=S_test)
        fairness_history.append(fairness)

        performance_history_profiles = calculate_acc_per_profile(learner, y_pred, y_test, profiles_test, performance_history_profiles)
        fairness_history_profiles = calculate_fairness_per_profile(learner, y_pred, y_test, S_test, profiles_test, fairness_history_profiles, fairness_metric)
        f1_scores_history_profiles = calculate_f1_per_profile(learner, y_pred, y_test, profiles_test, f1_scores_history_profiles)

        np.random.seed()
        query_idx = np.random.choice(X_train_pool.shape[0], size=n_instances, replace=False)
        query_inst = X_train_pool[query_idx]

        total_instances = total_instances + n_instances

        query_labels = y_train_pool[query_idx]

        X_train_pool = np.delete(X_train_pool, query_idx, axis=0)
        y_train_pool = np.delete(y_train_pool, query_idx, axis=0)

        learner.teach(query_inst, query_labels)

    return performance_history, fairness_history, f1_score_history, instances, performance_history_profiles, fairness_history_profiles, f1_scores_history_profiles

# Function to save the performance of sampling strategy
def plot_performance_random_profiles(dataset_name, df, y_label, S, n_experiments, profiles, fairness_metric, n_queries, random_number, set_seed=42, model_mode=LogisticRegression):
    performance_histories = {}
    fairness_histories = {}
    f1_scores_histories = {}

    performance_histories_profiles = {key: {} for key in np.unique(profiles)}
    fairness_histories_profiles = {key: {} for key in np.unique(profiles)}
    f1_scores_histories_profiles = {key: {} for key in np.unique(profiles)}

    f1_scores = []
    accuracies = []
    fairness = []

    random_numbers = [random_number]

    for i in range(n_experiments):
        print('experiment', i, random_numbers)
        performance_history, fairness_history, f1_score_history, instances, performance_history_profiles, fairness_history_profiles, f1_scores_history_profiles = active_learning_random_profiles(df.values, y_label.values, S.values, 8, 1, profiles, random_numbers[i], fairness_metric, n_queries=n_queries, model_mode=model_mode)
        accuracies.append(performance_history)
        fairness.append(fairness_history)
        f1_scores.append(f1_score_history)

        for j in range(len(performance_history)):
            if i == 0:
                performance_histories[j] = []
                performance_histories[j].append(performance_history[j])
            else:
                performance_histories[j].append(performance_history[j])
        for profile in performance_history_profiles:
            for iteratie in range(len(performance_history)):
                if i == 0:
                    performance_histories_profiles[profile][iteratie] = []
                    performance_histories_profiles[profile][iteratie].append(performance_history_profiles[profile][iteratie])
                else:
                    performance_histories_profiles[profile][iteratie].append(performance_history_profiles[profile][iteratie])

        for j in range(len(fairness_history)):
            if i == 0:
                fairness_histories[j] = []
                fairness_histories[j].append(fairness_history[j])
            else:
                fairness_histories[j].append(fairness_history[j])
        for profile in fairness_history_profiles:
            for iteratie in range(len(fairness_history)):
                if i == 0:
                    fairness_histories_profiles[profile][iteratie] = []
                    fairness_histories_profiles[profile][iteratie].append(fairness_history_profiles[profile][iteratie])
                else:
                    fairness_histories_profiles[profile][iteratie].append(fairairness_history_profiles[profile][iteratie])

        for j in range(len(f1_score_history)):
            if i == 0:
                f1_scores_histories[j] = []
                f1_scores_histories[j].append(f1_score_history[j])
            else:
                f1_scores_histories[j].append(f1_score_history[j])
        for profile in f1_scores_history_profiles:
            for iteratie in range(len(f1_score_history)):
                if i == 0:
                    f1_scores_histories_profiles[profile][iteratie] = []
                    f1_scores_histories_profiles[profile][iteratie].append(f1_scores_history_profiles[profile][iteratie])
                else:
                    f1_scores_histories_profiles[profile][iteratie].append(f1_scores_history_profiles[profile][iteratie])

    # save the results - Accuracy
    save_dict_to_csv(performance_histories_profiles, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_accuracy_histories_profiles_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')
    save_dict_to_csv(performance_histories, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_accuracy_histories_total_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')

    # save the results - Fairness
    save_dict_to_csv(fairness_histories_profiles, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_fairness_histories_profiles_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')
    save_dict_to_csv(fairness_histories, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_fairness_histories_total_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')

    # save the results - F1-score
    save_dict_to_csv(f1_scores_histories_profiles, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_f1_histories_profiles_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')
    save_dict_to_csv(f1_scores_histories, 'Results/' + str(dataset_name) + 'Random' + str(random_number) + 'experiments_f1_histories_total_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) + 'iter_' + '.csv')

    return

################### START OF PROGRAM ##########################
if __name__ == "__main__":
    start_time = time.time()
    print('argument', sys.argv)
    if len(sys.argv) >= 7:
        n_samples = int(sys.argv[1])
        dataset_path = str(sys.argv[2])
        target_value = str(sys.argv[3])
        sensitive_attr = str(sys.argv[4])
        subgroups = str(sys.argv[5])
        dataset_name = str(sys.argv[6])

        # Print a summary of the prompts
        print("\n")
        print("Summary of Input:")
        print(f"The program wil sample: {n_samples} samples")
        print(f"applied on the dataset: {dataset_path}")
        print(f"with as target_value: {target_value}")
        print(f"and sensitive_attr: {sensitive_attr}")
    else:
        print(f"Not enough command line arguments. Please provide all 6 prompts.")
        exit()
    print(f"\n")

    # Define own random number for reproduceablity, or use:
    random_number = random.randint(1, 100)

    print('START', random_number)
    file_path = os.path.abspath(dataset_path)
    df = pd.read_csv(file_path)

    y_label = df[target_value]
    df = df.drop(target_value, axis=1)

    S = df[sensitive_attr]
    df = df.drop(sensitive_attr, axis=1)
    profiles = df[subgroups]
    df = df.drop(subgroups, axis=1)

    ###########################
    # numbers = [377, 52, 346, 279, 44, 302, 216, 15, 47, 111, 119, 258, 13, 287, 101, 366, 332, 359, 214, 112]

    # for random_number in numbers:
    plot_performance_random_profiles(dataset_name, df, y_label, S, 1, profiles, demographic_parity_difference, n_samples, random_number, model_mode = RandomForestClassifier)
    
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Program {random_number} finished in {elapsed_time:.2f} seconds.")

