# This script include functions to initialise and train a simple neuron network
# Run this scrpit will print out the performance of the model trained under the specified hyper-parameters.

# import libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from utilities import import_data
from utilities import CreateDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from utilities import get_satisfaction
from utilities import compute_accuracy
from utilities import test
from utilities import set_seed
from utilities import import_data
from utilities import CreateDataset


# Define Model
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        # Define layers
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size), 
            nn.Sigmoid(), 
            nn.Linear(hidden_size, num_classes)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.squeeze(x)
        return x


# train the model (modified from lab 2)
def train(model, train_loader, num_epochs, optimiser, print_step = 50, verbose = False):
    should_print = lambda epoch : (verbose and ((epoch + 1) % print_step == 0 or epoch == 0)) # decide when to print
    model.train()
    criterion = nn.MSELoss()

    # train the model by batch
    for epoch in range(num_epochs):
        train_loss = 0 # total loss in this epoch
        
        for _, (x, y) in enumerate(train_loader):
            optimiser.zero_grad()  # zero the gradient buffer
            outputs = model(x) # forward pass

            loss = criterion(outputs, y.to(torch.float)) # calculate loss
            
            loss.backward() # backward pass
            optimiser.step() # update weights
            
            # calculate accuracy
            if (should_print(epoch)):
                train_loss += loss
        if (should_print(epoch)):
            # accuracy = correct/len(train_loader.dataset) * 100
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {(train_loss / len(train_loader)):.4f}')
    

    
def main():
    # Define hyperparameter
    input_size = 15
    output_size = 1 # real number
    
    hidden_size = 5
    num_epochs = 1000
    batch_size = 10
    learning_rate = 0.001
    weight_decay = 1e-5
    momemtum = 0.9
    

    # make results determinstic
    seed = None
    if seed != None:
        set_seed(seed)

    # import data
    data, _, input_size = import_data()

    # randomly split data into training set (80%) and testing set (20%)
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=seed)
    train_data, test_data = np.array(train_data), np.array(test_data)

    # setup data loader
    train_dataset = CreateDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initialise network
    simple_nn = SimpleNet(input_size, hidden_size, output_size)

    # define optimiser
    # optimiser = torch.optim.Adam(simple_nn.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimiser = torch.optim.SGD(simple_nn.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momemtum)

    # train the model
    train(simple_nn, train_loader, num_epochs, optimiser, print_step = 50, verbose = True)

    # test the model
    test(simple_nn, train_data, test_data)

if __name__ == "__main__":
    main()