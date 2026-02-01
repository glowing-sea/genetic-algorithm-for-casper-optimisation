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

# Load from the result of Grid Search instead of running it again
# This script may takes one to two days to complete
LOAD_RESULT = True

# Search Space (8^4 = 4096)
HYPERPARAMETERS_CANDIDATES = {
    'max_hidden_neurons': [1, 2, 3, 5, 7, 10, 13, 15],
    'P': [0.01, 0.05, 0.1, 0.5, 1, 5, 15, 20],
    'D': [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'L3': [0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.007, 0.01]
}

SEARCH_SPACE_SIZE = (len(HYPERPARAMETERS_CANDIDATES['max_hidden_neurons']) * 
                 len(HYPERPARAMETERS_CANDIDATES['P']) * 
                 len(HYPERPARAMETERS_CANDIDATES['D']) * 
                 len(HYPERPARAMETERS_CANDIDATES['L3'])) 

DEVICE = "cpu"
    
# make results determinstic
SEED = 4660
if SEED != None:
    set_seed(SEED)

DNA_SIZE = 12 

# Number of training for each hyperparameter combination
N_SPLITS = 10
N_REPEATS = 4

def chromosome_to_hyperparameters(chromosome):
    """
    This function accept a chromosome STRING or ARRAY and decrypt it into a hyperparameters directionary,
    which is the input of the Capser training function
    """
    
    # Ensure that the chromosome length is correct
    assert len(chromosome) == DNA_SIZE, f"Chromosome should be {DNA_SIZE} bits long!"

    if isinstance(chromosome, np.ndarray):
        chromosome = ''.join(map(str, chromosome)) # convert to string

    # Split chromosome into segments for each hyperparameter
    max_hidden_neurons_bits = chromosome[0:3]
    P_bits = chromosome[3:6]
    D_bits = chromosome[6:9]
    L3_bits = chromosome[9:12]

    # Convert bits to indices
    max_hidden_neurons_index = int(max_hidden_neurons_bits, 2)
    P_index = int(P_bits, 2)
    D_index = int(D_bits, 2)
    L3_index = int(L3_bits, 2)
    
    # Calculate L1 L2 L3
    L3 = HYPERPARAMETERS_CANDIDATES['L3'][L3_index]
    L2 = 5 * L3
    L1 = 200 * L3
    
    # Constructure directionary
    hyperparameters = {
        'max_hidden_neurons': HYPERPARAMETERS_CANDIDATES['max_hidden_neurons'][max_hidden_neurons_index],
        'P': HYPERPARAMETERS_CANDIDATES['P'][P_index],
        'D': HYPERPARAMETERS_CANDIDATES['D'][D_index],
        'lrs': [L1,L2,L3]
    }
    return hyperparameters

def int_to_12bit_binary(num):
    return bin(num)[2:].zfill(12)

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
    fitness_table.to_csv('results/fitness_table.csv')
    
def load_table():
    return pd.read_csv('results/fitness_table.csv', index_col=0, dtype={0: str})

def grid_search(train_data):
    """
    Compute the MSE for all 8 * 8 * 8 * 8 hyperparameter combinations and save it in 'results/fitness_table.csv'
    """
    fitness_table = create_table()
    start_time = time.time()
    for i in range(SEARCH_SPACE_SIZE):
        chromosome = int_to_12bit_binary(i)
        hyperparameters = chromosome_to_hyperparameters(chromosome)
        MSE = rkf_validator(train_data, hyperparameters, N_SPLITS, N_REPEATS, device=DEVICE, fast_mode=True, verbose=False)
        fitness_table.loc[chromosome] = MSE
        print(f"Searched Hyperparameter Combination: {i + 1}/{SEARCH_SPACE_SIZE}   MSE: {MSE}", end = '\r')
        # Save the progross every 64 search
        if (i + 1) % 64 == 0:
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
    print("The top 10 hyperparameter combinations (chromosomes) and their MSE")
    fitness_table = load_table()
    top_10s = fitness_table.sort_values(by='MSE', ascending=True).head(10)
    for i in range(len(top_10s)):
        chromosome = top_10s.index[i]
        print(top_10s.index[i], end = '\t')
        print(top_10s.iloc[i][0], end = '\t')
        print(chromosome_to_hyperparameters(chromosome))
    print()

    # Print the best hyperparameter combination
    best_chromosome = top_10s.index[0]
    print(f'Best Chromosome: {best_chromosome}')
    best_hyperparameters = chromosome_to_hyperparameters(best_chromosome)
    print(f'Best Hyperparameters: {best_hyperparameters}')
    print()

    # Give the best hyperparameter combination a final test
    final_test(data, best_hyperparameters, 10, 20, device=DEVICE)

if __name__ == "__main__":
    main()

