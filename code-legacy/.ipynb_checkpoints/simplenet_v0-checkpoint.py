# This script include functions to initialise and train a simple neuron network
# Run this scrpit will print out the performance of the model trained under the specified hyper-parameters.

# import libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from data_preprocessing import import_data
from data_preprocessing import split_data
from data_preprocessing import CreateDataset
from sklearn.metrics import confusion_matrix

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
        return x


# train the model (modified from lab 2)
def train(model, train_loader, num_epochs, optimiser, print_step = 50, verbose = False):
    should_print = lambda epoch : (verbose and ((epoch + 1) % print_step == 0 or epoch == 0)) # decide when to print
    model.train()
    criterion = nn.CrossEntropyLoss()

    # train the model by batch
    for epoch in range(num_epochs):
        correct = 0 # total correct predictions
        total_loss = 0 # total loss in this epoch
        
        for _, (x, y) in enumerate(train_loader):
            optimiser.zero_grad()  # zero the gradient buffer
            outputs = model(x) # forward pass
            loss = criterion(outputs, y) # calculate loss
            loss.backward() # backward pass
            optimiser.step() # update weights
            
            # calculate accuracy
            if (should_print(epoch)):
                _, predicted = torch.max(outputs, 1)
                correct += sum(predicted.cpu().data.numpy() == y.cpu().data.numpy())
                total_loss += loss
        if (should_print(epoch)):
            accuracy = correct/len(train_loader.dataset) * 100
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}')

# test the model (modified from lab 2)
def test(model, train_data, test_data):
    model.eval()
    
    # test on train set
    train_input = train_data.iloc[:, 1:]
    train_target = train_data.iloc[:, 0]
    inputs = torch.Tensor(train_input.values).float()
    targets = torch.Tensor(train_target.values).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    print("Confusion matrix for training:")
    print(confusion_matrix(targets.data, predicted.cpu().long().data))

    # test on test set
    test_input = test_data.iloc[:, 1:]
    test_target = test_data.iloc[:, 0]
    inputs = torch.Tensor(test_input.values).float()
    targets = torch.Tensor(test_target.values).long()
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    print("Confusion matrix for testing:")
    print(confusion_matrix(targets.data, predicted.cpu().long().data))
    
    # print test accuracy
    total = predicted.size(0)
    correct = predicted.cpu().data.numpy() == targets.data.numpy()
    print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))

def main():
    # Define hyperparameter
    input_size = None
    hidden_size = 50
    num_classes = None
    num_epochs = 500
    batch_size = 10
    learning_rate = 0.001


    # make results determinstic
    seed = None
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # import data
    data, num_classes, input_size = import_data()

    # randomly split data into training set (80%) and testing set (20%)
    train_data, test_data = split_data(data, seed = None)

    # setup data loader
    train_dataset = CreateDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # initialise network
    simple_nn = SimpleNet(input_size, hidden_size, num_classes)

    # define optimiser
    optimiser = torch.optim.Adam(simple_nn.parameters(), lr=learning_rate)

    # train the model
    train(simple_nn, train_loader, num_epochs, optimiser, print_step = 50, verbose = True)

    # test the model
    test(simple_nn, train_data, test_data)

if __name__ == "__main__":
    main()