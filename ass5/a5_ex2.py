from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from a4_ex1 import SimpleNetwork
from dataset import get_dataset



def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    eval_data: torch.utils.data.Dataset,
    num_epochs: int,
    show_progress: bool = False
    ) -> tuple[list, list]:

    #my pseudo code first attempt:
    # create the dataset i could use
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = DataLoader(eval_data, batch_size=64, shuffle=True)

    # optimizer and loss 
    opt = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss = nn.MSELoss()

    # training loop
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_pbar = tqdm.tqdm(range(num_epochs))
    counter = 0

    for epoch in epoch_pbar:
        batch_train_loss = []
        network.train()
        batch_pbar = tqdm.tqdm(dataloader) if show_progress else dataloader
        for minibatch in batch_pbar:

            x, Y = minibatch
            Y = torch.unsqueeze(Y, dim=1)

            pred = network(x)
            mini_loss = loss(pred, Y)

            opt.zero_grad()
            mini_loss.backward()
            opt.step()
            
            batch_train_loss.append(round(mini_loss.item(),3))

            if show_progress:
                batch_pbar.set_postfix(loss=mini_loss.item())
                batch_pbar.update(1)




        network.eval()
        batch_val_loss = []
        with torch.no_grad():
            for minibatch in validation_loader:

                x, Y = minibatch
                Y = torch.unsqueeze(Y, dim=1)

                pred = network(x)
                mini_loss = loss(pred, Y)

                batch_val_loss.append(round(mini_loss.item(),3))
        
        epoch_train_loss.append(np.average(batch_train_loss))
        epoch_val_loss.append(np.average(batch_val_loss))

        epoch_pbar.set_postfix(train_loss=epoch_train_loss[epoch], eval_loss=epoch_val_loss[epoch])
        tqdm.tqdm.write(f"Epoch {epoch} --- train_loss: {epoch_train_loss[epoch]:.2f} \
                        --- eval_loss: {epoch_val_loss[epoch]:.2f}")
        

        if epoch >= 1:
            if epoch_val_loss[-1] > epoch_val_loss[-2]:
                counter += 1
            else:
                counter = 0

            if counter == 3:
                break
        else: 
            pass

    
    return (epoch_train_loss, epoch_val_loss)


def plot_losses(train_losses: list, eval_losses: list):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, color="blue", label="train loss")
    plt.plot(epochs, eval_losses, color="red", label="eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.show()

    



if __name__ == "__main__":
    # create the network itselfs
    fnn_network = SimpleNetwork(input_neurons=32, hidden_neurons=32, output_neurons=1)
    (train, eval) = get_dataset()

    epoch_train_loss, epoch_val_loss = training_loop(network=fnn_network, train_data=train, 
                                        eval_data=eval, num_epochs=5, show_progress=True)
    

    plot_losses(epoch_train_loss, epoch_val_loss)
