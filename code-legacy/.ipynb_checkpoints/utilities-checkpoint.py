# This script contains some utility functions

import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Convert model output to an satisfication interger ranging from 1 to 7
def get_satisfaction(outputs):
    satisfaction = []
    for output in outputs:
        if output > 7:
            output = 7
        elif output < 1:
            output = 1
        else:
            output = torch.round(output)
            output = output.detach().numpy()
        satisfaction.append(output)
    return np.array(satisfaction)

def compute_accuracy(outputs, targets, tolerance):
    outputs = get_satisfaction(outputs)
    total = len(outputs)
    correct = 0
    for i in range(len(outputs)):
        if abs(outputs[i] - targets[i]) <= tolerance:
            correct += 1
    return correct / total * 100

# test the model
def test(model, train_data, test_data):
    model.eval()
    MSE = nn.MSELoss()
    MAE = nn.L1Loss()
    
    train_input = train_data[:, 1:]
    train_target = train_data[:, 0]
    inputs = torch.Tensor(train_input).float()
    targets = torch.Tensor(train_target).float()
    outputs = model(inputs)
    
    MS_loss_train = MSE(outputs, targets)
    MA_loss_train = MAE(outputs, targets)
    
    acc0 = compute_accuracy(outputs, targets, 0)
    acc1 = compute_accuracy(outputs, targets, 1)
    
    print(f'\nTraining Set: MSE = {MS_loss_train} MAE = {MA_loss_train}')
    print(f'Strict Acc = {acc0:.2f} Loose Acc = {acc1:.2f}')
    print(confusion_matrix(targets.data, get_satisfaction(outputs)).T)
    

    test_input = test_data[:, 1:]
    test_target = test_data[:, 0]
    inputs = torch.Tensor(test_input).float()
    targets = torch.Tensor(test_target).float()
    outputs = model(inputs)
    
    MS_loss_test = MSE(outputs, targets)
    MA_loss_test = MAE(outputs, targets)
    
    acc0 = compute_accuracy(outputs, targets, 0)
    acc1 = compute_accuracy(outputs, targets, 1)
    
    print(f'\nTesting Set: MSE = {MS_loss_test} MAE = {MA_loss_test}')
    print(f'Strict Acc = {acc0:.2f} Loose Acc = {acc1:.2f}')
    print(confusion_matrix(targets.data, get_satisfaction(outputs)).T)


# test the model (modified from lab 2)
def test_classification(model, train_data, test_data):
    model.eval()
    
    # test on train set
    train_input = train_data[:, 1:]
    train_target = train_data[:, 0]
    inputs = torch.Tensor(train_input).float()
    targets = torch.Tensor(train_target).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    print("Confusion matrix for training:")
    print(confusion_matrix(targets.data, predicted.cpu().long().data))

    # test on test set
    test_input = test_data[:, 1:]
    test_target = test_data[:, 0]
    inputs = torch.Tensor(test_input).float()
    targets = torch.Tensor(test_target).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    print("Confusion matrix for testing:")
    print(confusion_matrix(targets.data, predicted.cpu().long().data))
    
    # print test accuracy
    total = predicted.size(0)
    correct = predicted.cpu().data.numpy() == targets.data.numpy()
    print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))

# set a random seed
def set_seed(seed):
    random.seed(seed)
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

# split data into train and test set
def split_data(data, seed = None):
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=seed)
    return train_data, test_data
    
# define a customise torch dataset (modified from Lab 2)
class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data_tensor = torch.tensor(data, dtype=torch.float)

    # a function to get items by index
    def __getitem__(self, index):
        input = self.data_tensor[index][1:]
        target = self.data_tensor[index][0].long()

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n

