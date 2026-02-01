# This script contains some utility functions

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from imblearn.over_sampling import SMOTE

# set a random seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def print_params(model):
    for p in model.parameters():
        is_frozen = "Frozen " if not p.requires_grad else "Trainable "
        print(f'{is_frozen} {p.data} {p.data.shape}')
        print()

# import data from the csv file
def import_data():
    try:
        data = pd.read_csv('data/snippets.csv')
        num_of_labels = len(data.iloc[:,0].unique())
        num_of_features = data.shape[1] - 1
        return data, num_of_labels, num_of_features
    except:
        print("Please run 'data_preprocess.py' to preprocess the data!")
        return None, None, None
    
# Take a data set and oversample it
# if a class with samples < 6, use random oversampling
# if a calss with samples >= 6, use SMOTE
# the target sample size of each class = the sample size of the largest class
def custom_smote(dataset, verbose):
    # Extract labels and features
    y = dataset[:, 0]
    X = dataset[:, 1:]
    
    # Get unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    desired_samples = np.max(counts)

    if verbose:
        print("Before Oversampling")
        print(unique_classes)
        print(counts)

    X_resampled_list = []
    y_resampled_list = []

    # Classes with 6 samples or more, use SMOTE
    mask_large_classes = np.isin(y, [cls for cls, count in zip(unique_classes, counts) if count >= 6])
    X_large = X[mask_large_classes]
    y_large = y[mask_large_classes]
    
    # Run SMOTE
    smote = SMOTE(sampling_strategy = "auto")
    X_resampled_large, y_resampled_large = smote.fit_resample(X_large, y_large)
    
    X_resampled_list.append(X_resampled_large)
    y_resampled_list.append(y_resampled_large)

    # Classes with leas than 6 samples, use random oversampling
    for cls, sample_count in zip(unique_classes, counts):
        if sample_count < 6:
            mask = y == cls
            X_cls = X[mask]
            y_cls = y[mask]
        
            # Random oversampling
            num_to_add = desired_samples - sample_count
            samples_to_add_X = X_cls[np.random.choice(X_cls.shape[0], num_to_add)]
            samples_to_add_y = np.full(num_to_add, cls)
        
            X_resampled_cls = np.vstack([X_cls, samples_to_add_X])
            y_resampled_cls = np.hstack([y_cls, samples_to_add_y])
        
            X_resampled_list.append(X_resampled_cls)
            y_resampled_list.append(y_resampled_cls)

    # Concatenate the resampled datasets
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.hstack(y_resampled_list)
    resampled_dataset = np.column_stack((y_resampled, X_resampled))
    np.random.shuffle(resampled_dataset)

    if verbose:
        unique_labels, counts = np.unique(resampled_dataset[:, 0], return_counts=True)
        print("After Oversampling")
        print(unique_labels)
        print(counts)
    
    return resampled_dataset

