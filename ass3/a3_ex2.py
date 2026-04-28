import numpy as np
import os
import glob
import torch
from torch.utils.data import DataLoader

from a3_ex1 import RandomImagePixelationDataset


def stack_with_padding(batch_as_list: list):

    pixel_height_list, pixel_width_list = [], []
    target_list, abspath_list = [], []
    for single4tuple in batch_as_list:
        # read in all the tuple entries
        pixelated_image = single4tuple[0]
        known_array = single4tuple[1]
        target_array = single4tuple[2]
        abspath = single4tuple[3]


        # first lets get the lists for the full pixelated images sorted out
        # not we can use same numbers for known array so we dont track this again for known array
        pixel_height = pixelated_image.shape[1]
        pixel_width = pixelated_image.shape[2]

        pixel_height_list.append(pixel_height)
        pixel_width_list.append(pixel_width)

        # finally append target into list
        # similarly get abspath into list
        target_list.append(target_array)
        abspath_list.append(abspath)


    
    pixel_mheight = max(pixel_height_list)
    pixel_mwidth = max(pixel_width_list)



    #somehow do padding
    padded_pixel_list, padded_known_list = [], []
    for single4tuple in batch_as_list:
        pixelated_image = single4tuple[0]
        known_array = single4tuple[1] 

        pixel_height = pixelated_image.shape[1]
        pixel_width = pixelated_image.shape[2]

        padded_pixel_image = np.pad(pixelated_image, [(0, 0), (0, pixel_mheight-pixel_height), \
                            (0, pixel_mwidth-pixel_width)], mode='constant', constant_values=0)


        padded_known_array = np.pad(known_array, [(0, 0), (0, pixel_mheight-pixel_height), \
        (0, pixel_mwidth-pixel_width)], mode='constant', constant_values=1)



        padded_pixel_list.append(padded_pixel_image)
        padded_known_list.append(padded_known_array)


    stacked_padded_pixel_images = torch.tensor(np.stack(padded_pixel_list, axis=0))
    stacked_known_arrays = torch.tensor(np.stack(padded_known_list, axis=0))

    torch_targets = [torch.from_numpy(i) for i in target_list]

    return (stacked_padded_pixel_images, stacked_known_arrays, torch_targets, abspath_list)

    

if __name__ == "__main__":
    image_dir = "./ass2/grey_images"

    if os.path.isdir(image_dir):
        image_list = sorted(glob.glob(image_dir + "/**/*.jpg" ,recursive=True))


    ds = RandomImagePixelationDataset(image_dir=image_dir, width_range=(50,300), 
                                    height_range=(50,300), size_range=(10,50))



    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=stack_with_padding)
    for batch in dl:
        print(batch[0].shape)  # should be (2,1,H_max,W_max)
        break