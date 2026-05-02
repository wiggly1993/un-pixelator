import glob
import numpy as np
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import tqdm


from a3_ex1 import RandomImagePixelationDataset
from a3_ex2 import stack_with_padding
from a6_ex1 import UNet



def training_loop(
    network: torch.nn.Module,
    train_data: torch.utils.data.Dataset,
    eval_data: torch.utils.data.Dataset,
    num_epochs: int,
    show_progress: bool = False
    ) -> tuple[list, list]:

    # create the dataset i could use
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=stack_with_padding)
    validation_loader = DataLoader(eval_data, batch_size=16, shuffle=True, collate_fn=stack_with_padding)

    # optimizer and loss 
    opt = torch.optim.Adam(network.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    loss = nn.MSELoss()

    # send to device 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    network = network.to(device)

    # prepare for training loop
    epoch_train_loss = []
    epoch_val_loss = []
    epoch_pbar = tqdm.tqdm(range(num_epochs))
    counter = 0

    for epoch in epoch_pbar:
        batch_train_loss = []
        train_batch_pbar = tqdm.tqdm(train_loader) if show_progress else train_loader
        network.train()

        for minibatch in train_batch_pbar:
            stacked_padded_pixel_images, stacked_known_arrays, torch_targets, _ = minibatch

            # stack the input
            prepped_input = torch.cat([stacked_padded_pixel_images, stacked_known_arrays], dim=1).float()
            prepped_input = prepped_input.to(device)

            raw_output = network(prepped_input)

            single_losses = []
            for i in range(len(torch_targets)):
                predicted_pixels = raw_output[i, 0][~stacked_known_arrays[i, 0].bool()]
                target = torch_targets[i].float().flatten()
                target = target.to(device)

                l = loss(predicted_pixels, target)
                single_losses.append(l)


            batch_loss = sum(single_losses) / len(torch_targets)

            opt.zero_grad() 
            batch_loss.backward()
            opt.step()


            batch_train_loss.append(round(batch_loss.item(),3))


        
        network.eval()
        batch_val_loss = []
        val_batch_pbar = tqdm.tqdm(validation_loader) if show_progress else val_loader
        with torch.no_grad():
            for minibatch in val_batch_pbar:
                stacked_padded_pixel_images, stacked_known_arrays, torch_targets, _ = minibatch

                # stack the input
                prepped_input = torch.cat([stacked_padded_pixel_images, stacked_known_arrays], dim=1).float()
                prepped_input = prepped_input.to(device)

                raw_output = network(prepped_input)

                single_losses = []
                for i in range(len(torch_targets)):
                    predicted_pixels = raw_output[i, 0][~stacked_known_arrays[i, 0].bool()]
                    target = torch_targets[i].float().flatten()
                    target = target.to(device)

                    l = loss(predicted_pixels, target)
                    single_losses.append(l)


                batch_loss = sum(single_losses) / len(torch_targets)
                batch_val_loss.append(round(batch_loss.item(),3))

        
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


        # if len(epoch_val_loss) > 1 and epoch_val_loss[-1] < min(epoch_val_loss[:-1]):
        #     torch.save(network.state_dict(), "./ass6/best_model.pth")
                

    return (epoch_train_loss, epoch_val_loss)




if __name__ == "__main__":
    path = "./ass6/grey_training"
    dir_list = glob.glob(path + "/**/*.jpg" ,recursive=True)

    full_ds = RandomImagePixelationDataset(image_dir=path, width_range=(4,32), 
                                             height_range=(4,32), size_range=(4,16))
    
    train_data, val_data = random_split(full_ds, [0.85, 0.15])

    # returns in total (stacked_padded_pixel_images, stacked_known_arrays, torch_targets, abspath_list)
    # where stacked_padded_pixel_images has shape (N, 1, H, W)
    # train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=stack_with_padding)
    # val_loader = DataLoader(val_data, batch_size=4, shuffle=False, collate_fn=stack_with_padding)
    # baby_loader = DataLoader(baby_data, batch_size=4, shuffle=True, collate_fn=stack_with_padding)

    Unet_model = UNet(num_classes=1)
    

    (epoch_train_loss, epoch_val_loss) = training_loop(network=Unet_model, 
    train_data=train_data, eval_data=val_data, num_epochs=5, show_progress=True)

    
    








