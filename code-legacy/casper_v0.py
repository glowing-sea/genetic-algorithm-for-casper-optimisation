# import libraries
import random
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from sklearn.model_selection import train_test_split
from utilities import import_data
from utilities import CreateDataset
from utilities import test
from utilities import set_seed

class CasperNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons=0):
        super(CasperNet, self).__init__()
        self.total_neurons = input_dim + output_dim
        self.output_dim = output_dim
        self.hidden_dim = input_dim # dimension of the hidden layer (dimension of the cascased input)
        self.output_layer = nn.Linear(input_dim, output_dim) # start with minimal network
        self.hidden_layers = nn.ModuleList() # maintain a list of hidden layer
        self.sigmoid = nn.Sigmoid()

        
        for i in range(num_hidden_neurons):
            self.add_neuron()

    def forward(self, x):
        
        # loop through all hidden layer, cascade to inputs
        for layer in self.hidden_layers:
            hidden_output = self.sigmoid(layer(x)) # output of a hidden unit
            x = torch.cat((x, hidden_output), dim=1)  # cascade the output to the previous inputs
        
        x = self.output_layer(x)
        x = torch.squeeze(x)

        return x

    
    def add_neuron(self):
        new_neuron = nn.Linear(self.hidden_dim, 1)
        torch.nn.init.xavier_uniform_(new_neuron.weight)
        self.hidden_dim += 1 # update hidden layer dimension
        self.total_neurons += 1
        self.hidden_layers.append(new_neuron) # add the candidate to the hidden unit list
        
        
        # Preserve old weights and biases
        old_weights = self.output_layer.weight.data
        old_biases = self.output_layer.bias.data
    
        # Create new output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
    
        # Assign old weights and biases back
        self.output_layer.weight.data[:, :-1] = old_weights
        self.output_layer.bias.data = old_biases
        
        # Initialize the new weights (last column) - using xavier initialization as an example
        nn.init.xavier_uniform_(self.output_layer.weight.data[:, -1].unsqueeze(1))

def train_network(model, train_data, optimiser, P, verbose=False):
    max_epoch = 5000 # just in case it never stop
    threshold = 0.01
    criterion = nn.MSELoss()
    epoch = 1
    prev_train_loss = float('inf')
    train_loss = 0
    
    threshold_P = 0 # how many times does the model fall below the threshold
    threshold_P_max = 15 + model.total_neurons * P # stop training if threshold_P exceed this value
    
    while True:   
        
        train_input = train_data[:, 1:]
        train_target = train_data[:, 0]
        inputs = torch.Tensor(train_input).float()
        labels = torch.Tensor(train_target).float()
        
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        train_loss = loss.item()
        
        
        if prev_train_loss == float('inf'):
            percent = 1
        elif prev_train_loss == 0:
            break
        else:
            percent = abs((prev_train_loss - train_loss) / prev_train_loss)
        
        # print loss and accuracy
        if verbose:
            if (epoch % 50) == 0 or epoch - 1 == 0:
                print(f'Epoch {epoch}, Loss: {train_loss :.4f}')
        
        
        if percent < threshold:
            threshold_P += 1
        else:
            threshold_P = 0
        
        # prevent infinite loop
        if epoch >= max_epoch:
            break 
        
        if threshold_P >= threshold_P_max:
            break                          
        
        prev_train_loss = train_loss
        epoch += 1
    if verbose:
        print(f'Training stop at epoch {epoch} with loss = {train_loss:.4f}')
    return train_loss


def train(model, train_data, max_hidden_neurons, P, D, T, lr, verbose=False):
    
    # define optimiser parameter
    def update_optimiser(model):
        optimiser_parameters = get_optimiser_parameters(model, lr[0], lr[1], lr[2])
        return CasperRprop(optimiser_parameters,D=D,T=T)
    
    hidden_neurons = 0
    
    # keep adding neuron to the network
    while(True):
        # train the network
        optimiser = update_optimiser(model)
        loss = train_network(model, train_data, optimiser, P, verbose)
        
        
        if hidden_neurons >= max_hidden_neurons:
            break
        
        # add an neuron
        model.add_neuron()
        hidden_neurons += 1
    
    return model


class CasperRprop(optim.Rprop):
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), D=0.01, T=1, *args, **kwargs):
        super(CasperRprop, self).__init__(params, lr, etas, step_sizes, *args, **kwargs)
        
        # Additional parameters for weight decay
        self.D = D
        self.T = T
        self.H_epoch = 0  # Initialize epoch count
        
    def step(self, closure=None):
        # Increment epoch count
        self.H_epoch += 1
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Your custom weight decay term
                weight_decay_term = - self.D * torch.sign(p.data) * p.data**2 * (2**(-self.T * self.H_epoch))
                
                # Apply the weight decay to the gradient
                grad.add_(weight_decay_term)
                
                # Rest of the Rprop logic remains the same...

        # Call the parent's step method to apply the modified gradient
        super(CasperRprop, self).step(closure)
        
        return loss
    
def get_optimiser_parameters(model, lr_l1, lr_l2, lr_l3):
    param_l1 = []
    param_l3 = []
    
    param_l2 = [list(model.output_layer.parameters())[0]]
    param_l3.append(list(model.output_layer.parameters())[1]) # bias of output neurons is L3
    
    if len(model.hidden_layers) != 0:
        param_l1 = [list(model.hidden_layers[-1].parameters())[0]] # weights of the new hidden neuron is L1
        param_l3.extend(list(model.hidden_layers[:-1].parameters())) # weights and bias of the other hidden neurons are L3
        param_l3.append(list(model.hidden_layers[-1].parameters())[1]) # bias of the new hidden neuron is L3
        

    
    # Create parameter groups
    parameters = [
        {"params": param_l1, "lr": lr_l1},
        {"params": param_l2, "lr": lr_l2},
        {"params": param_l3, "lr": lr_l3},
    ]
    
    return parameters

def main():
    
    # make results determinstic
    seed = None
    if seed != None:
        set_seed(seed)
        
    # Define hyperparameter
    hidden_neurons = 1
    P = 0.1
    lr = [0.001, 0.005, 0.2]
    D = 0
    T = 1
    
    input_size = 15
    output_size = 1
    
    # import data
    data, _, input_size = import_data()

    # randomly split data into training set (80%) and testing set (20%)
    train_data, test_data, _, _ = train_test_split(data, data.iloc[:,0], test_size=0.2, random_state=seed)
    train_data, test_data = np.array(train_data), np.array(test_data)

    # initialise network
    casper_net = CasperNet(input_size, output_size)

    # train the model
    train(casper_net, train_data, hidden_neurons, P, D, T, lr, verbose=True)

    # test the model
    test(casper_net, train_data, test_data)

if __name__ == "__main__":
    main()