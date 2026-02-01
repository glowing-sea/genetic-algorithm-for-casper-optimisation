# This class is the implementation of Casper
# Other script my want to call the train() function direction

# Acknowledgement
# ChatGPT has been used for debugging and previded ideas of writing

# import libraries
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

from utilities import custom_smote

class CasperNet(nn.Module):
    """
    Define the model
    """

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
        if(x.shape[1] == 1):
            x = torch.squeeze(x)

        return x

    def add_neuron(self):
        """
        Add a new hidden layer and expend the output layer weight matrix
        """

        # Determine the device of the current model
        device = next(self.parameters()).device

        new_neuron = nn.Linear(self.hidden_dim, 1)
        torch.nn.init.xavier_uniform_(new_neuron.weight)
        self.hidden_dim += 1 # update hidden layer dimension
        self.total_neurons += 1
        self.hidden_layers.append(new_neuron) # add the candidate to the hidden unit list

        # Move the new neuron to the device
        new_neuron = new_neuron.to(device)
        
        # Preserve old weights and biases
        old_weights = self.output_layer.weight.data
        old_biases = self.output_layer.bias.data
    
        # Create new output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
    
        # Move the updated output layer to the device
        self.output_layer = self.output_layer.to(device)

        # Assign old weights and biases back
        self.output_layer.weight.data[:, :-1] = old_weights
        self.output_layer.bias.data = old_biases
        
        # Initialize the new weights (last column) - using xavier initialization as an example
        nn.init.xavier_uniform_(self.output_layer.weight.data[:, -1].unsqueeze(1))


class CasperRprop(optim.Optimizer):
    """
    Define the Casper Rprop Optimiser
    """

    def __init__(self, params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50), D=0.005, T=0.01, H_epoch=1):
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes, D=D, T=T, H_epoch=H_epoch)
        super(CasperRprop, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        # loop through each parameter group
        for group in self.param_groups:
            # loop through each weight matrix in a group
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad # obtain the gradient of the weight matrix
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step_size'] = torch.full_like(p, group['lr']) # the lr for each entry in the matrix to be either L1/L2/L3
                    state['prev_grad'] = torch.zeros_like(p) # initialise the previous gradient for each entry as 0
                
                # obtain step size and previous gradient for each weight matrix's entries
                step_size = state['step_size']
                prev_grad = state['prev_grad']

                # Adjust the step size based on the sign of (grad * prev_grad)
                sign = grad.mul(prev_grad).sign()
                step_increase = torch.where(sign > 0, group['etas'][1], 1.)
                step_decrease = torch.where(sign < 0, group['etas'][0], 1.)
                step_size.mul_(step_increase * step_decrease) # etas[1] if > 0, etas[0] if < 0, 1 otherwise
                step_size = torch.clamp(step_size, group['step_sizes'][0], group['step_sizes'][1])

                # Weight decay term applied to gradient
                weight_decay = -group['D'] * torch.sign(p.data) * p.data**2 * (2**(-group['T'] * group['H_epoch']))
                grad.add_(weight_decay)

                # Weight add or substract step_size from the weight based on the sign of the weight
                weight_update = torch.where(grad > 0, -step_size, torch.where(grad < 0, step_size, 0))
                p.add_(weight_update)
                
                # Store the current gradient for the next iteration
                state['prev_grad'] = grad

        return None


def get_optimiser_parameters(model, lrs):
    """
    A method of assigning different learning rates to different parts of the model
    """
    param_l1 = []
    param_l3 = []
    
    param_l2 = [list(model.output_layer.parameters())[0]] # weights of output neurons is L2
    param_l3.append(list(model.output_layer.parameters())[1]) # bias of output neurons is L3
    
    if len(model.hidden_layers) != 0:
        param_l1 = [list(model.hidden_layers[-1].parameters())[0]] # weights of the new hidden neuron is L1
        param_l3.extend(list(model.hidden_layers[:-1].parameters())) # weights and bias of the other hidden neurons are L3
        param_l3.append(list(model.hidden_layers[-1].parameters())[1]) # bias of the new hidden neuron is L3
        
    # Create parameter groups
    parameters = [
        {"params": param_l1, "lr": lrs[0]},
        {"params": param_l2, "lr": lrs[1]},
        {"params": param_l3, "lr": lrs[2]},
    ]
    
    return parameters


def train_network(model, train_data, optimiser, P, task, device, verbose):
    """
    train the network after adding new hidden neurons
    """
    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    max_epoch = 5000 # just in case it never stop
    threshold = 0.01
    epoch = 1
    prev_train_loss = float('inf')
    train_loss = 0
    
    threshold_P = 0 # how many times does the model fall below the threshold
    threshold_P_max = 15 + model.total_neurons * P # stop training if threshold_P exceed this value

    train_input = train_data[:, 1:]
    train_target = train_data[:, 0]
    inputs = torch.Tensor(train_input).float().to(device)
    if task == 'regression':
        labels = torch.Tensor(train_target).float().to(device)
    else:
        labels = torch.Tensor(train_target).long().to(device)
    
    # train the model through a number of epochs
    while True:   
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


def train(model, train_data, hyperparameters, task = 'regression', verbose = False):
    """
    train the network (keep adding new hidden neurons and call train_network)
    """
    # Determine device
    device = next(model.parameters()).device
    if verbose:
        print(f"Training with {device}")

    # Oversample data
    train_data = custom_smote(train_data, verbose)

    # Obtain hyperparameter
    max_hidden_neurons = hyperparameters['max_hidden_neurons']
    P = hyperparameters['P']
    D = hyperparameters['D']
    lrs = hyperparameters['lrs']

  
    # define optimiser parameter
    def update_optimiser(model):
        optimiser_parameters = get_optimiser_parameters(model, lrs)
        return CasperRprop(optimiser_parameters,D=D)
    
    hidden_neurons = 0
    
    # keep adding neuron to the network
    while(True):
        # train the network
        optimiser = update_optimiser(model)
        train_network(model, train_data, optimiser, P, task, device, verbose)
        
        if hidden_neurons >= max_hidden_neurons:
            break
        
        # add an neuron
        model.add_neuron()
        hidden_neurons += 1
    
    return model