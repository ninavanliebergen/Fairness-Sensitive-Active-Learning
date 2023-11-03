from pandas import read_csv, to_datetime, get_dummies, concat
import pandas as pd
import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling

import matplotlib.pyplot as plt
import matplotlib as mpl

import random

from modAL.uncertainty import uncertainty_sampling, classifier_entropy, classifier_margin, classifier_uncertainty, entropy_sampling, margin_sampling, uncertainty_sampling
from modAL.models import Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling

import sys

import time
import csv

import fairlearn
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference, equalized_odds_ratio

import concurrent.futures
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import multiprocessing
from multiprocessing import Process, Value, Array

from collections import defaultdict

# import all the helpers functions:
from helpers import *

import multiprocessing

def process(chunk_id, chunk):
    fairness_per_instances = {}

    chunk_time = time.time()
    for i in range(0, len(chunk)):
        model_time = time.time()
        fairness_per_instances[chunk_id+i] = 0
        
        # 2 labels! 0 en 1!
        for j in range(0, 2):
            Xl_tmp=np.append(Xl_global,[chunk[i]],axis=0)
            yl_tmp=np.append(yl_global,[j],axis=0)

            # train een nieuw model!!
            model_tmp = RandomForestClassifier()
                
            model_tmp.fit(Xl_tmp, yl_tmp)
            y_pred = model_tmp.predict(X_test_global)
            
            unfairness_tmp = fairness_metric_global(y_test_global,
                            y_pred,
                            sensitive_features=S_test_global)

            # calculate the propability of this unfairness
            # y = model_tmp.predict([chunk[i]])[0] #the most probable label
            p = model_tmp.predict_proba([chunk[i]])[0][j] #p for label y=j
            # print('unfairness_tmp:', unfairness_tmp, 'y:', y, 'p:',p)

            fairness_tmp = (unfairness_tmp)*p

            fairness_per_instances[chunk_id+i] += fairness_tmp
        # print('time to check 1 instance', time.time()-model_time)
    best = min(fairness_per_instances, key=fairness_per_instances.get)
    if chunk_id == 5:
        print('time to check 1 chunk (chunk 5)', time.time()-chunk_time)
    return {best: fairness_per_instances[best]} 


def FAL_sampling_parallel(Xl, Xu, yl, X_test, S_test, y_test, fairness_metric):
    FAL_sampling_parallel_time = time.time()
    num_workers = n_chunks

    ## CHECK DIT HIERONDER!!!
    chunk_size = len(Xu) // n_chunks
    print('LEN CHUNCK', chunk_size, len(Xu), len(Xu)%chunk_size)

    Xu_chunks = split(Xu, n_chunks)

    global Xl_global
    Xl_global = Xl

    global yl_global, X_test_global, S_test_global, y_test_global, fairness_metric_global
    yl_global = yl

    X_test_global = X_test
    S_test_global = S_test
    y_test_global = y_test
    fairness_metric_global = fairness_metric


    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(process, [(i, chunk) for i, chunk in Xu_chunks])

    print("All workers are done.")

    combined_results = {k: v for d in results for k, v in d.items()}
    query_idx = min(combined_results, key=combined_results.get)
    query_inst = Xu[query_idx]
    
    print('FAL_sampling_parallel_time', FAL_sampling_parallel_time-time.time())
    return [query_idx], [query_inst]

def save_dict_to_csv(dictionary, filename):
    save_dict_to_csv_time = time.time()
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(dictionary.keys())
        writer.writerow(dictionary.values())
    print('save_dict_to_csv_time', save_dict_to_csv_time-time.time())

# Function for Fair Active Learning
def active_learning_FAL(X, y, S, n_start, n_instances, profiles, seed, fairness_metric, n_queries=500, model_mode=LogisticRegression):
    performance_history_profiles = {key: [] for key in np.unique(profiles)}
    fairness_history_profiles = {key: [] for key in np.unique(profiles)}
    f1_scores_history_profiles = {key: [] for key in np.unique(profiles)}
    
    performance_history = []
    fairness_history = []
    f1_score_history = []

    instances = []

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test, S_train, S_test, profiles_train, profiles_test = train_test_split(X, y, S, profiles, test_size=0.3, random_state=seed)

    # START WITH x SAMPLES
    np.random.seed(seed)  # Set a seed value for reproducibility
    random_indices = np.random.choice(X_train.shape[0], size=n_start, replace=False)
    print(random_indices)
    X_train_start = X_train[random_indices]
    y_train_start = y_train[random_indices]
    S_train_start = S_train[random_indices]
    print('y_train_start',y_train_start )

    X_labeled = X_train_start
    y_labeled = y_train_start


    # THE POOL WHERE NO LABEL IS KNOWN
    X_train_pool = np.delete(X_train, random_indices, axis=0)
    y_train_pool = np.delete(y_train, random_indices, axis=0)
    S_train_pool = np.delete(S_train, random_indices, axis=0)

    # Create an ActiveLearner with a linear regression model
    learner = ActiveLearner(
        estimator=RandomForestClassifier(), 
        X_training=X_train_start,
        y_training=y_train_start
    )  

    total_instances = n_start

    # Iterate for x steps, selecting and incorporating new samples
    for _ in range(n_queries):
        time_start = time.time()
        print('query', _)
        # Calculate accuracy on the current model
        y_pred = learner.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        performance_history.append(accuracy)
        instances.append(total_instances)
        
        # Calculate F1 score on the current model
        f1 = f1_score(y_test, y_pred)
        f1_score_history.append(f1)
        
        # Calculate Fairness (here demographic parity)
        fairness = fairness_metric(y_test,
                                    y_pred,
                                    sensitive_features=S_test)
        fairness_history.append(fairness)

        performance_history_profiles = calculate_acc_per_profile(learner, y_pred, y_test, profiles_test, performance_history_profiles)
        fairness_history_profiles = calculate_fairness_per_profile(learner, y_pred, y_test, S_test, profiles_test, fairness_history_profiles, fairness_metric)
        f1_scores_history_profiles  = calculate_f1_per_profile(learner, y_pred, y_test, profiles_test, f1_scores_history_profiles)

        # choose next datasample
        query_idx, query_inst = FAL_sampling_parallel(X_labeled, X_train_pool, y_labeled, X_test, S_test, y_test, fairness_metric)
        total_instances = total_instances + n_instances

        # Obtain the true labels for the queried instances
        query_labels = y_train_pool[query_idx]

        # add to the new labeled pool
        X_labeled = np.append(X_labeled, X_train_pool[query_idx], axis=0)
        y_labeled = np.append(y_labeled, y_train_pool[query_idx], axis=0)

        # Remove the queried instance from the unlabeled pool.
        X_train_pool = np.delete(X_train_pool, query_idx, axis=0)
        y_train_pool = np.delete(y_train_pool, query_idx, axis=0)

        # Teach the learner the new instances
        learner.teach(query_inst, query_labels)
        time_end = time.time()
        ttime = time_start - time_end
        print(f"query {_} finished in {ttime:.2f} seconds.")

    return performance_history, fairness_history, f1_score_history, instances, performance_history_profiles, fairness_history_profiles, f1_scores_history_profiles

def plot_performance_FAL_profiles(dataset_name, df, y_label, S, n_experiments, profiles, fairness_metric, set_seed=42, n_queries=40, model_mode=LogisticRegression):
    performance_histories = {}
    fairness_histories = {}
    f1_scores_histories = {}
    
    performance_histories_profiles = {key: {} for key in np.unique(profiles)}
    fairness_histories_profiles = {key: {} for key in np.unique(profiles)}
    f1_scores_histories_profiles = {key: {} for key in np.unique(profiles)}
    
    f1_scores = []
    accuracies = []
    fairness = []
    
    random.seed(set_seed)  # Set a seed value for reproducibility
    random_numbers = random.sample(range(401), k=n_experiments)

    for i in range(n_experiments):
        print('experiment:', i)
        performance_history, fairness_history, f1_score_history, instances, performance_history_profiles, fairness_history_profiles, f1_scores_history_profiles = \
                                active_learning_FAL(df.values, y_label.values, S.values, 8, 1, \
                                                            profiles, random_numbers[i], fairness_metric, \
                                                            n_queries=n_queries, model_mode=model_mode)
        accuracies.append(performance_history)
        fairness.append(fairness_history)
        f1_scores.append(f1_score_history)
        
        # Accuracy
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
        
        ## FAIRNESS
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
                    fairness_histories_profiles[profile][iteratie].append(fairness_history_profiles[profile][iteratie])
        
        # F1-score
        for j in range(len(f1_score_history)):
            if i == 0:
                f1_scores_histories[j] = []
                f1_scores_histories[j].append(f1_score_history[j])
            else:
                f1_scores_histories[j].append(f1_score_history[j])
        for profile in performance_history_profiles:
            for iteratie in range(len(performance_history)):
                if i == 0:
                    f1_scores_histories_profiles[profile][iteratie] = []
                    f1_scores_histories_profiles[profile][iteratie].append(fairness_history_profiles[profile][iteratie])
                else:
                    f1_scores_histories_profiles[profile][iteratie].append(fairness_history_profiles[profile][iteratie])

 

    # save the results - Accuracy
    save_dict_to_csv(performance_histories_profiles, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_accuracy_histories_profiles_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')
    save_dict_to_csv(performance_histories, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_accuracy_histories_total_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')

    # save the results - Fairness
    save_dict_to_csv(fairness_histories_profiles, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_fairness_histories_profiles_' + str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')
    save_dict_to_csv(fairness_histories, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_fairness_histories_total_' +  str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')

    # save the results - F1-score
    save_dict_to_csv(f1_scores_histories_profiles, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_f1_histories_profiles_' +  str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')
    save_dict_to_csv(f1_scores_histories, 'Results/' + str(dataset_name) + 'FAL_total_LG_' + str(random_number) + 'experiments_f1_histories_total_' +  str(len(np.unique(profiles))) + 'profiles' + str(n_queries) +'iter_'+'.csv')

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
    
    # Define own random number for reproduceablity, or use:
    random_number = random.randint(1, 100)

    file_path = os.path.abspath(dataset_path)
    df = pd.read_csv(file_path)

    # If the code is only applyied on a suppart of the dataset
    if len(sys.argv) > 7:
        subpart = float(sys.argv[7])
        # Calculate the number of elements for a % slice
        list_of_rows = df.values.tolist()
        slice_size = int(len(df) * subpart)

        # Take a random subpart slice of the list of rows
        random.seed(42) # set seed for always taking the same subpart
        random_slice = random.sample(list_of_rows, slice_size)

        # Create a new DataFrame with the sampled rows
        df = pd.DataFrame(random_slice, columns=df.columns)
        print(f"A subset of the dataset is use, a random sample of: {subpart}%")

    # For determining the amount of workers
    if len(sys.argv) > 8:
        n_chunks = int(sys.argv[8])
        print(f"The amount of workers is: {n_chunks}")
    # no amount of workers? Run serial: 1 worker
    else:
        print(f"The program will not run in parallel")
        n_chunks = 1
    print("\n")

    print('START', random_number)

    y_label = df[target_value]
    df = df.drop(target_value, axis=1)

    S = df[sensitive_attr]
    df = df.drop(sensitive_attr, axis=1)
    profiles = df[subgroups]
    df = df.drop(subgroups, axis=1)

    ###########################
    plot_performance_FAL_profiles(dataset_name, df, y_label, S, 1, profiles, demographic_parity_difference, set_seed=random_number, n_queries=n_samples, model_mode=RandomForestClassifier)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Program {random_number} finished in {elapsed_time:.2f} seconds.")
