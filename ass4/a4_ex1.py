import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self,
                 input_neurons: int,
                 hidden_neurons: int,
                 output_neurons: int,
                 activation_function: nn.Module = nn.ReLU()):
        super().__init__()

        self.Lin1 = nn.Linear(input_neurons, hidden_neurons, bias=True, device=None, dtype=None)
        self.Lin2 = nn.Linear(hidden_neurons, hidden_neurons, bias=True, device=None, dtype=None)
        self.Lin3 = nn.Linear(hidden_neurons, hidden_neurons, bias=True, device=None, dtype=None)
        self.Lin4 = nn.Linear(hidden_neurons, output_neurons, bias=True, device=None, dtype=None)

        self.activation = activation_function


    def forward(self, x):

        x = self.Lin1(x)
        x = self.activation(x)

        x = self.Lin2(x)
        x = self.activation(x)

        x = self.Lin3(x)
        x = self.activation(x)

        x = self.Lin4(x)
        
        return x