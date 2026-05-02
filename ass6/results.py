import torch
import matplotlib.pyplot as plt

from a3_ex1 import RandomImagePixelationDataset
from a6_ex1 import UNet


if __name__ == "__main__":

    grey_images_path = "./ass6/grey_training_later"
    saved_model_path = "./ass6/best_model.pth"

    full_ds = RandomImagePixelationDataset(image_dir=grey_images_path, width_range=(4,32), 
                                             height_range=(4,32), size_range=(4,16))
    
    #get a single sample from fullds
    #remember we need to turn this manually into tensors and squeeze
    #since we are no longer using tensors
    pixelated_image, known_array, target_array, image_file = full_ds[5]
    prepped_input = torch.cat([torch.tensor(pixelated_image), 
    torch.tensor(known_array)], dim=0).float().unsqueeze(0)
   


    trained_CNN = UNet(num_classes=1)
    trained_CNN.load_state_dict(torch.load(saved_model_path, weights_only=True))
    trained_CNN.eval()

    output = trained_CNN(prepped_input)
    print(output.shape)


    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(pixelated_image[0], cmap="gray")
    axes[0].set_title("Pixelated Input")
    axes[0].axis("off")

    axes[1].imshow(output[0, 0].detach().numpy(), cmap="gray")
    axes[1].set_title("Model Output (De-pixelated)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
