# This class contains test and validating function for Casper
# Running the main of this script will train the model on the Snippet dataset using a specific
# hyperparameter combination and random seed and print out the test result

# Acknowledgement
# ChatGPT has been used for debugging and previded ideas of writing

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from casper import CasperNet
from casper import train
from utilities import import_data
from utilities import set_seed

def test(model, data, verbose = False):
    """
    test the model (for regression task only)

    return: the average MSE annMAE, all unique labels, classwise average MSE, classwise average MAE
    of a model trained by a specific hypermarater combination and random seed
    """
    model.eval()
    device = next(model.parameters()).device
    
    MSE = nn.MSELoss(reduction='none')
    MAE = nn.L1Loss(reduction='none')
    
    inputs = torch.Tensor(data[:, 1:]).float().to(device)
    labels = torch.Tensor(data[:, 0]).float().to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        all_MSE = MSE(outputs, labels)
        all_MAE = MAE(outputs, labels)

        # Overall average MSE and L1 loss
        avg_MSE = torch.mean(all_MSE).cpu().item()
        avg_MAE = torch.mean(all_MAE).cpu().item()

        classwise_MSE = []
        classwise_MAE = []
        unique_labels = torch.unique(labels).cpu().numpy()

        # Class-wise MSE and L1 loss
        for cls in unique_labels:
            mask = (labels == cls)
            classwise_MSE.append(torch.mean(all_MSE[mask]).cpu().item())
            classwise_MAE.append(torch.mean(all_MAE[mask]).cpu().item())
    
    if verbose:
        print(f'Overall Training MSE: {avg_MSE}')
        print(f'Overall Training MAE: {avg_MAE}')
        print(unique_labels)
        print(classwise_MSE)
        print(classwise_MAE)

    return avg_MSE, avg_MAE, unique_labels, classwise_MSE, classwise_MAE

def test_fast(model, data):
    """
    A fast version of the test function that only return MSE
    return: the average MSE of a model trained by a specific hypermarater combination and random seed
    """
    model.eval()
    device = next(model.parameters()).device
    
    MSE = nn.MSELoss(reduction='none')
    
    inputs = torch.Tensor(data[:, 1:]).float().to(device)
    labels = torch.Tensor(data[:, 0]).float().to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        
        MSE = MSE(outputs, labels)

    return torch.mean(MSE).cpu().item()


def rkf_validator(data, hyperparameters, n_splits, n_repeats, device='cpu', fast_mode=True, verbose=False, oversampling=False):

    """
    validate a single hyperparameter combination (for regression task only)

    if fast_mode == true, only return the average MSE over all (n_splits * n_repeats) model
    if fast_mode == false, return:
    (1) a list of MSE (length = n_splits * n_repeats)
    (2) a list of MAE (length = n_splits * n_repeats)
    (3) a list of classwise_MSE (length = n_splits * n_repeats)
    (4) a list of classwise_MAE (length = n_splits * n_repeats)
    where each element is the MAE/MSE score of a model trained by the same hypermarater combination but different ramdom seed
    """
    
    data = np.array(data)
    input_dim = data.shape[1] - 1
    MSEs = []
    MAEs = []
    classwise_MSEs = []
    classwise_MAEs = []
    unique_labels = [] # not all test set contain all label so we need to record which label we have tested

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    
    # train the model n_splits * n_repeats times
    for i, (train_index, test_index) in enumerate(rkf.split(data)):
        # obtain training and validation set
        data_train, data_val = data[train_index], data[test_index]
            
        # initialise network
        casper_net = CasperNet(input_dim, 1).to(device)
            
        # train the model
        train(casper_net, data_train, hyperparameters, task="regression", verbose=False, oversampling=oversampling)

        # evaluate the model
        if fast_mode:
            MSE_score = test_fast(casper_net, data_val)
            MSEs.append(MSE_score)
            if verbose:
                print(f'Fold: {i + 1}/{(n_splits * n_repeats)}   MSE: {MSE_score:.4f}', end = '\r')
        else:
            MSE_score, MAE_score, unique_label, classwise_MSE, classwise_MAE = test(casper_net, data_val, verbose=False)
            unique_labels.append(unique_label)
            MSEs.append(MSE_score)
            MAEs.append(MAE_score)
            classwise_MSEs.append(classwise_MSE)
            classwise_MAEs.append(classwise_MAE)
            if verbose:
                print(f'Fold: {i + 1}/{(n_splits * n_repeats)}  MSE: {MSE_score:.4f}    MAE: {MAE_score:.4f}', end = '\r')
        

    if fast_mode:
        return np.mean(MSEs)
    else:
        return MSEs, MAEs, unique_labels, classwise_MSEs, classwise_MAEs
    
def final_test(data, hyperparameters, n_splits, n_repeats, device='cpu', oversampling=False):
    """
    train and test the model over (n_splits * n_repeats) runs with different train-test splits 
    and model initialisation
    print the result
    """
    num_of_labels = len(data.iloc[:,0].unique())
    
    print(f"Model is trained {n_splits * n_repeats} times")
    print("with each run has different test-training split and initialised weights")
    
    MSEs, L1Es, unique_labels, classwise_MSEs, classwise_L1Es = rkf_validator(data, 
                                                                              hyperparameters, 
                                                                              n_splits, n_repeats, 
                                                                              device=device, fast_mode=False,
                                                                              verbose=True, 
                                                                              oversampling=oversampling)
    
    print(f'Mean: {np.mean(MSEs)}')
    print(f'Median: {np.median(MSEs)}')
    print(f'Standard Deviation: {np.std(MSEs)}')
    print()
    print(f'Mean: {np.mean(L1Es)}')
    print(f'Median: {np.median(L1Es)}')
    print(f'Standard Deviation: {np.std(L1Es)}')

    plt.figure(figsize=(10,3))
    plt.subplot(1,2,1)
    plt.title("Distribution of Squared Errors")
    plt.hist(MSEs, color='lightgreen', ec='black', bins=20)
    plt.subplot(1,2,2)
    plt.title("Distribution of Absolute Errors")
    plt.hist(L1Es, color='lightgreen', ec='black', bins=20)
    plt.show()
    
    # now classwise_MSEs/classwise_MAEs is approximate the shape of (num_runs, num_classes)
    # where num_runs = n_splits * n_repeats and num_classes in this case is 7
    # turn it into the shape of (num_classes, num_runs)
    
    classwise_MSEs_T = [[], [], [], [], [], [], []]
    classwise_L1Es_T = [[], [], [], [], [], [], []]

    # calculate the MSE and L1E for each class
    for i in range(1,num_of_labels+1):
        for j in range(n_splits * n_repeats):     
            if i in unique_labels[j]:
                index = int(np.where(unique_labels[j] == i)[0][0])                
                classwise_MSEs_T[i-1].append(classwise_MSEs[j][index])
                classwise_L1Es_T[i-1].append(classwise_L1Es[j][index])
    
    # print the MSE and L1E for each class
    for i in range(1,num_of_labels+1):
        class_i_MSEs = classwise_MSEs_T[i-1]
        class_i_L1Es = classwise_L1Es_T[i-1]
        num_run = len(class_i_MSEs)
        print(f"Class {i}")
        print(f"Number of runs that contain MSE/MAE for this class: {num_run}")
        print(f"MSE: average over all {num_run} runs: {np.mean(class_i_MSEs):.4f}")
        print(f"MSE: median over all {num_run} runs: {np.median(class_i_MSEs):.4f}")
        print(f"MSE: std over all {num_run} runs: {np.std(class_i_MSEs):.4f}")
        print(f"L1E: average of MAE over all {num_run} runs: {np.mean(class_i_L1Es):.4f}")
        print(f"L1E: median over all {num_run} runs: {np.median(class_i_L1Es):.4f}")
        print(f"L1E std over all {num_run} runs: {np.std(class_i_L1Es):.4f}")
        print()

def main():
    """
    Show one example of training and testing
    """

    # select training device
    device = "cpu"
    
    # make results determinstic
    seed = 4650
    if seed != None:
        set_seed(seed)
        
    # Define hyperparameter
    hyperparameters = {'max_hidden_neurons': 15,
                  'P': 20,
                  'D': 0.005,
                  'lrs': [0.2, 0.005, 0.001]}
    
    input_size = 15
    output_size = 1
    
    # import data
    data, _, input_size = import_data()

    # randomly split data into training set (80%) and testing set (20%)
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=seed)
    train_data, test_data = np.array(train_data), np.array(test_data)

    # initialise network
    casper_net = CasperNet(input_size, output_size).to(device)

    # train the model
    train(casper_net, train_data, hyperparameters, task='regression', verbose=True)

    test(casper_net, test_data, verbose=True)

if __name__ == "__main__":
    main()