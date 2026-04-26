from PIL import Image
import numpy as np
import os

def to_grayscale(pil_image: np.ndarray = None):
    # check for whether shape is okey
    if len(pil_image.shape) not in [2,3]:
            print(f"shape is somehow totally fucked: {ValueError}")
            raise ValueError

    else:  
        if len(pil_image.shape) == 3:
            if pil_image.shape[2] == 3:
                    pass
            else:
                    print(f"The image has 3 dimensions (H,W,3) but the 3 is not there! {ValueError}")
                    raise ValueError
                
        if len(pil_image.shape) == 2:
            pil_image = np.expand_dims(pil_image, axis=0).copy()
            return pil_image

        else:
            pass
        
    
    original_dtype = pil_image.dtype
    copy_image = pil_image.copy()
    n_copy = copy_image/255

    # Clinear per channel
    C_linear = np.where(n_copy <= 0.04045, n_copy / 12.92, ((n_copy + 0.055) / 1.055) ** 2.4)

    # Ylinear (only valid for 3D case)
    Y_linear = 0.2126 * C_linear[..., 0] + 0.7152 * C_linear[..., 1] + 0.0722 * C_linear[..., 2]

    # Y
    Y = np.where(Y_linear <= 0.0031308, 12.92 * Y_linear, 1.055 * Y_linear ** (1/2.4) - 0.055)

    # denormalize
    Y = 255*Y

    if np.issubdtype(pil_image.dtype, np.integer):
         Y = np.round(Y,1)

    # cast original dtype
    Y = Y.astype(original_dtype)

    # expand dims
    Y = np.expand_dims(Y, axis=0)

    return Y
        
    



        
        
               







if __name__ == "__main__":
    path = "./ass2/test_images"
    dir_list = os.listdir(path)

    # print("pause")
    # print(os.path.join(path, dir_list[0]))

    # test_img = Image.open(os.path.join(path,dir_list[0]))
    # test_img = np.array(test_img)
    # res = to_grayscale(test_img)

    for iter, img in enumerate(dir_list):
         image = np.array(Image.open(os.path.join(path, img)))
         greyed_image = to_grayscale(image)
         #need to squeeze (1, H, W) into (H, W) 
         greyed_image = greyed_image.squeeze()

         prepared_for_saving = Image.fromarray(greyed_image)
         prepared_for_saving.save(f"./ass2/grey_images/{iter}_greyed.jpg")
