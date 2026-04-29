import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, 
                hidden_channels: int,
                num_hidden_layers: int,
                use_batchnormalization: bool,
                num_classes: int,
                kernel_size: int = 3,
                activation_function: nn.Module = nn.ReLU()):
        
        super().__init__() #CNN, self

        self.sequential_pipe = nn.Sequential()

        self.activation = activation_function
        batchnorm = nn.BatchNorm2d(hidden_channels)

        padding = kernel_size//2

        # 1st convolutional layer
        conv1 = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, 
                                kernel_size=kernel_size, padding=padding)
     

        self.sequential_pipe.append(conv1)
        self.sequential_pipe.append(self.activation)
        if use_batchnormalization:

             self.sequential_pipe.append(batchnorm)


        for i in range(num_hidden_layers):
            conv_hidden = nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, 
                    kernel_size=kernel_size, padding=padding)
            
            batchnorm = nn.BatchNorm2d(hidden_channels)
                     

            self.sequential_pipe.append(conv_hidden)
            self.sequential_pipe.append(self.activation)
            if use_batchnormalization:
                self.sequential_pipe.append(batchnorm)
            

        self.sequential_pipe.append(nn.Flatten())

        self.fc1 = nn.Linear(in_features=64*64*hidden_channels, out_features=num_classes)

        self.sequential_pipe.append(self.fc1)
       


                

    def forward(self, x):
        x = self.sequential_pipe(x)
        return x


if __name__ == "__main__":
    SimpleCNN_obj = SimpleCNN(input_channels=3, 
                            hidden_channels=32, 
                            num_hidden_layers=3, 
                            use_batchnormalization=True, 
                            num_classes=10)
    

    x = torch.randn(5, 3, 64, 64)

    output = SimpleCNN_obj(x)
    print(output)