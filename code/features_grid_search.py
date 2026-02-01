"""
This script finds the best hyperparameter combination using Grid Search

While running, it also builds up a one-to-one mapping between all possible 2^12 hyperparameter combiniations 
(encoded as Chromosoomes) and saved in 'results/fitness_table.csv'

This script may takes one to two days to complete, if LOAD_RESULT, load the results 'results/fitness_table.csv'
directly from my last run.

"""

import numpy as np
import time
import pandas as pd
from utilities import import_data
from utilities import set_seed
from casper_test import final_test
from casper_test import rkf_validator
from sklearn.model_selection import train_test_split
from hypers_grid_search import chromosome_to_hyperparameters

# Load from the result of Grid Search instead of running it again
# This script may takes one to two days to complete
LOAD_RESULT = True

# Choose the best hyperparameter combination from the hyperparameter searches and fix it
TOP_10S = pd.read_csv('results/top_10s.csv', index_col=0, dtype={0: str})


HYPERPARAMETERS = chromosome_to_hyperparameters(TOP_10S.index[0])
print(f'Hyperpararmeters Used: {HYPERPARAMETERS}')

SEARCH_SPACE_SIZE = 2 ** 15

DEVICE = "cpu"
    
# make results determinstic
SEED = 4660
if SEED != None:
    set_seed(SEED)

DNA_SIZE =15

# Number of training for each hyperparameter combination
N_SPLITS = 10
N_REPEATS = 4

def chromosome_to_dataset(chromosome, dataset):
    """
    Convert a chromosome into a dataset with a subset of the features
    Returns a dataset with the subset of features based on the chromosome
    """
    if isinstance(chromosome, str):
        # Convert the binary string to a list of integers
        feature_indices = [int(bit) for bit in chromosome]
    elif isinstance(chromosome, (list, np.ndarray)):
        feature_indices = chromosome
    else:
        raise ValueError("Chromosome should be either a 15 length binary string or a 15 length number binary array")
    
    if len(feature_indices) != 15:
        raise ValueError("Chromosome length should be 15")
    
    # Get the columns to keep based on the chromosome
    columns_to_keep = [dataset.columns[0]] + [dataset.columns[i+1] for i, bit in enumerate(feature_indices) if bit]
    
    return dataset[columns_to_keep]

def int_to_15bit_binary(num):
    return bin(num)[2:].zfill(15)

def create_table():
    chromosomes = []
    MSEs = []

    fitness_table = pd.DataFrame({
    'Chromosome': chromosomes,
    'MSE': MSEs
    })
    fitness_table.set_index('Chromosome', inplace=True)
    return fitness_table

def save_table(fitness_table):
    fitness_table.to_csv('results/feature_selection_fitness.csv')

def load_table():
    return pd.read_csv('results/feature_selection_fitness.csv', index_col=0, dtype={0: str})

def grid_search(train_data):
    """
    Compute the MSE for models training on all 2^12 possible subsets of features and save it in 'results/feature_selection_fitness.csv'
    """
    fitness_table = create_table()
    start_time = time.time()

    # Give infinite MSE to empty set
    fitness_table.loc[int_to_15bit_binary(0)] = np.inf

    for i in range(1, 32):
        chromosome = int_to_15bit_binary(i)
        train_data_subset = chromosome_to_dataset(chromosome, train_data)
        MSE = rkf_validator(train_data_subset, HYPERPARAMETERS, N_SPLITS, N_REPEATS, device=DEVICE, fast_mode=True, verbose=False)
        fitness_table.loc[chromosome] = MSE
        print(f"Searched Feature Subset: {i + 1}/{SEARCH_SPACE_SIZE}   MSE: {MSE}", end = '\r')
        # Save the progross every 64 search
        if (i + 1) % 32 == 0:
            save_table(fitness_table)
    print()
    print(f"Grid Search Finished, Spent: {time.time() - start_time}s")

def main():
    # import data
    data, _, _ = import_data()
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=SEED)

    if not LOAD_RESULT:
        grid_search(train_data)


    # Print the top 10 chromosome
    print("The top 10 feature subset and their MSE")
    fitness_table = load_table()
    top_10s = fitness_table.sort_values(by='MSE', ascending=True).head(10)
    for i in range(len(top_10s)):
        print(top_10s.index[i], end = '\n')
        print(chromosome_to_dataset(top_10s.index[i], data).columns[1:], end = '\n')
        print(top_10s.iloc[i][0], end = '\n')
    print()

    # Print the best hyperparameter combination
    best_chromosome = top_10s.index[0]
    print(f'Best Feature Subset: {best_chromosome}')
    print(chromosome_to_dataset(best_chromosome, data).columns[1:])
    print()

    # Give the best hyperparameter combination a final test
    final_test(chromosome_to_dataset(best_chromosome, data), HYPERPARAMETERS, 10, 20, device=DEVICE)

if __name__ == "__main__":
    main()