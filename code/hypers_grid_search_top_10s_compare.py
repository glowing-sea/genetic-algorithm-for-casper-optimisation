"""
This script recompute and reorder the top 10 chromosome by evaluating them through more powerful n repeated k fold

store the result in 'results/top_10s.csv'

"""

from utilities import import_data
from utilities import set_seed
from casper_test import rkf_validator
from sklearn.model_selection import train_test_split

from hypers_grid_search import load_table
from hypers_grid_search import chromosome_to_hyperparameters

DEVICE = "cpu"
    
# make results determinstic
SEED = 4660
if SEED != None:
    set_seed(SEED)

DNA_SIZ = 12 

# Number of training for each hyperparameter combination
N_SPLITS = 10
N_REPEATS = 50

FITNESS_TABLE = load_table()


def main():
    # import data
    data, _, _ = import_data()
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=SEED)

    top_10s = FITNESS_TABLE.sort_values(by='MSE', ascending=True).head(10)

    for i in range(10):
        chromosome = top_10s.index[i]
        hyperparameters = chromosome_to_hyperparameters(chromosome)
        MSE = rkf_validator(train_data, hyperparameters, N_SPLITS, N_REPEATS, device=DEVICE, fast_mode=True, verbose=False)
        top_10s.loc[chromosome] = MSE
        print(f"Searched Hyperparameter Combination: {i + 1}/{10}   MSE: {MSE}", end = '\r')
    print()

    # Print the top 10 chromosome
    print("The top 10 hyperparameter combinations (chromosomes) and their MSE")

    top_10s = top_10s.sort_values(by='MSE', ascending=True).head(10)
    top_10s.to_csv('results/top_10s.csv')
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

if __name__ == "__main__":
    main()

