import numpy as np
import os
from PIL import Image
import math

def prepare_image(
    image: np.ndarray = None,
    x: int  = None,
    y: int  = None,
    width: int  = None,
    height: int  = None,
    size: int  = None
    ):

    invalid_operation = image.shape[0] != 1 or image.shape[0] != 1 \
        or len(image.shape) != 3 or width < 2 or height < 2 or size < 2 \
        or x + width > image.shape[1] or y + height > image.shape[2] or x < 0 or y < 0
    
    if invalid_operation:
        raise ValueError
    
    pixelated_image = image.copy()

    for height_step in range(math.ceil(height/size)):
        for width_step in range(math.ceil(width/size)):
            y_start = y + size * height_step
            y_end = min(y + size * (height_step + 1), y + height)
            x_start = x + size * width_step
            x_end = min(x + size * (width_step + 1), x + width)
            
            block = pixelated_image[0, y_start:y_end, x_start:x_end]
            avg = block.mean()
            pixelated_image[0, y_start:y_end, x_start:x_end] = avg



    # full image in bool version, where true = original pixels, false = pixelated area
    known_array = np.ones_like(image)
    known_array = known_array.astype(bool)
    known_array[0, y:y+height, x:x+width] = 0

    target_array = image[0, y:y+height, x:x+width].copy()

    
    return pixelated_image, known_array, target_array

     

    



if __name__ == "__main__":
    path = "./ass2/grey_images"
    dir_list = os.listdir(path)

    image = Image.open(os.path.join(path,dir_list[0]))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    print(f"the shape of the image is {image.shape}")
    prepare_image(image=image, x=1, y=1, width=300, height=300, size=30)