from torch.utils.data import Dataset
from typing import Optional
import os
import glob
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt


from a2_ex1 import to_grayscale
from a2_ex2 import prepare_image


class RandomImagePixelationDataset(Dataset):
    def __init__(self,
    image_dir,
    width_range: tuple[int, int] = None,
    height_range: tuple[int, int] = None,
    size_range: tuple[int, int] = None,
    dtype: Optional[type] = None):
        
        self.dtype = dtype
        self.size_range = size_range
        self.height_range = height_range
        self.width_range = width_range
           
        image_dir = os.path.abspath(image_dir)

        if os.path.isdir(image_dir):
            self.image_list = sorted(glob.glob(image_dir + "/**/*.jpg" ,recursive=True))


        min_value_invalid = width_range[0]<2 or height_range[0]<2 or size_range[0]<2 \
        or width_range[0]>width_range[1] or height_range[0]>height_range[1] or \
        size_range[0]>size_range[1]

        if min_value_invalid:
            print(f"check your ranges either one is below 2 or \
                  the max value is smaller than the min value: {ValueError}")
            raise ValueError



        
    def __getitem__(self, index):
    
        img = Image.open(self.image_list[index])

        img = np.array(img, dtype=self.dtype)
        img = to_grayscale(img)

        random.seed(index)

        height = np.shape(img)[1]
        width = np.shape(img)[2]

        rwidth = random.randint(self.width_range[0], self.width_range[1])
        rheight = random.randint(self.height_range[0], self.height_range[1])

        if rwidth > width:
            rwidth = width

        if rheight > height:
            rheight = height

        rx = random.randint(0, width-rwidth)
        ry = random.randint(0, height-rheight)

        rsize = random.randint(self.size_range[0], self.size_range[1])

        pixelated_image, known_array, target_array = \
            prepare_image(image=img, x=rx,y=ry, width=rwidth, height=rheight, size=rsize)

        return (pixelated_image, known_array, target_array, self.image_list[index])
    

    def __len__(self):
        return len(self.image_list)



        



if __name__ == "__main__":
    image_dir = "./ass2/grey_images"
    ds = RandomImagePixelationDataset(image_dir=image_dir, width_range=(50,300), 
                                             height_range=(50,300), size_range=(10,50))
    


    for pixelated_image, known_array, target_array, image_file in ds:
        fig, axes = plt.subplots(ncols=3)
        axes[0].imshow(pixelated_image[0], cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("pixelated_image")
        axes[1].imshow(known_array[0], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("known_array")
        axes[2].imshow(target_array[0], cmap="gray", vmin=0, vmax=255)
        axes[2].set_title("target_array")
        fig.suptitle(image_file)
        fig.tight_layout()
        plt.show()
        plt.close()