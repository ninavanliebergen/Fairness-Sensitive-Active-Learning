# Helpers Functions

import time
import csv
from sklearn.metrics import accuracy_score, f1_score

# Function to split a list 'a' into 'n' equal-sized parts
def split(a, n):
    k, m = divmod(len(a), n)
    return ((i, a[i*k+min(i, m):(i+1)*k+min(i+1, m)]) for i in range(n))

# Function to save a dictionary to a CSV file
def save_dict_to_csv(dictionary, filename):
    save_dict_to_csv_time = time.time()
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(dictionary.keys())
        writer.writerow(dictionary.values())
    print('save_dict_to_csv_time', save_dict_to_csv_time - time.time())

# Function to create a dictionary of data per profile
def profile_data_dict(profiles_test, X_train_pool):
    profile_test_data = {}
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_test_data:
            profile_test_data[profile_idx] = []
        profile_test_data[profile_idx].append(X_train_pool[i])
    return profile_test_data

# Function to create a dictionary of sensitive attributes per profile
def profile_S_dict(profiles_test, S_train_pool):
    profile_test_S = {}
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_test_S:
            profile_test_S[profile_idx] = []
        profile_test_S[profile_idx].append(S_train_pool[i])
    return profile_test_S

# Function to create a dictionary of labels per profile
def profile_labels_dict(profiles_test, y_train_pool):
    profile_labels = {}
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_labels:
            profile_labels[profile_idx] = []
        profile_labels[profile_idx].append(y_train_pool[i])
    return profile_labels

# Function to calculate accuracy per profile
def calculate_acc_per_profile(learner, y_pred, y_test, profiles_test, performance_history_profiles):
    profile_y_dict = {}
    profile_p_dict = {}
 
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_y_dict:
            profile_y_dict[profile_idx] = []
            profile_p_dict[profile_idx] = []
        profile_y_dict[profile_idx].append(y_test[i])
        profile_p_dict[profile_idx].append(y_pred[i])

    for profile in profiles_test.unique():
        acc = accuracy_score(profile_y_dict[profile], profile_p_dict[profile])
        
        if profile not in performance_history_profiles:
            performance_history_profiles[profile] = []
        performance_history_profiles[profile].append(acc)
    
    return performance_history_profiles

# Function to calculate F1 score per profile
def calculate_f1_per_profile(learner, y_pred, y_test, profiles_test, f1_scores_history_profiles):
    profile_y_dict = {}
    profile_p_dict = {}
 
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_y_dict:
            profile_y_dict[profile_idx] = []
            profile_p_dict[profile_idx] = []
        profile_y_dict[profile_idx].append(y_test[i])
        profile_p_dict[profile_idx].append(y_pred[i])

    for profile in profiles_test.unique():
        f1 = f1_score(profile_y_dict[profile], profile_p_dict[profile], zero_division=1.0)
        if profile not in f1_scores_history_profiles:
            f1_scores_history_profiles[profile] = []
        f1_scores_history_profiles[profile].append(f1)
    return f1_scores_history_profiles

# Function to calculate fairness metric per profile
def calculate_fairness_per_profile(learner, y_pred, y_test, S_test, profiles_test, fairness_history_profiles, fairness_metric):
    profile_S_dict = {}
    profile_y_dict = {}
    profile_p_dict = {}
 
    for i, profile_idx in enumerate(profiles_test):
        if profile_idx not in profile_S_dict:
            profile_S_dict[profile_idx] = []
            profile_y_dict[profile_idx] = []
            profile_p_dict[profile_idx] = []
        profile_S_dict[profile_idx].append(S_test[i])
        profile_y_dict[profile_idx].append(y_test[i])
        profile_p_dict[profile_idx].append(y_pred[i])

    for profile in profiles_test.unique():
        fairness = fairness_metric(profile_y_dict[profile], profile_p_dict[profile], sensitive_features=profile_S_dict[profile])
        
        if profile not in fairness_history_profiles:
            fairness_history_profiles[profile] = []
        fairness_history_profiles[profile].append(fairness)
    
    return fairness_history_profiles
