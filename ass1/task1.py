from PIL import Image
import os
import numpy as np
import hashlib
import shutil
import glob



def validate_images(input_dir: str, output_dir: str, log_file: str = None, formatter: str = "07d"):

    ## input_dir checks
    # first we turn relative path into absolute path
    input_dir = os.path.abspath(input_dir)
    # next check whether the path is actually a valid path and not some bs
    if os.path.isdir(input_dir):
        # check recursively for all files within this path folder and all subfolders
        # additionally sort them ascending 
        dir_list = sorted(glob.glob(input_dir + "/**/*", recursive=True))
        #print(dir_list)
    else:
        print(f"The path is invalid !!")
        raise ValueError

    # checking if the directory demo_folder exist or not.

    if not os.path.exists(output_dir):
        print(f"output_dir does not exist...creating the folder now")
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(output_dir)
        print(f"output folder {output_dir} created and ready to go")
    
    else:
        print(f"Output folder exists already and ready to go")


    seen_hashes = set()
    number_of_conversions = 0
    # open .log file to write all failures inside of it
    with open(log_file, "w") as f:
        for img in dir_list:
            # check for correct endings
            if img.endswith(".jpg") or img.endswith(".JPG") or img.endswith(".jpeg") or img.endswith(".JPEG"):
                # check for file size
                if os.path.getsize(img) < 250000:
                    # opening the images in PIL allows us to check the colors and shape (after converting into numpy)
                    try:
                        with Image.open(img) as temp_image:
                            valid_shape = ((temp_image.mode == "RGB" 
                                and np.array(temp_image).shape[0] > 100 
                                and np.array(temp_image).shape[1] > 100 
                                and np.array(temp_image).shape[2] == 3) or   
                                (temp_image.mode == "L" 
                                and np.array(temp_image).shape[0] > 100 
                                and np.array(temp_image).shape[1] > 100)
                                and np.var(np.array(temp_image)) > 0)
                            
                    except Exception as e:
                        valid_shape = False
                        print(e)
                        f.write(f"Image: {os.path.basename(os.path.normpath(img))} can not be opened with PIL!! \n" )
                        continue
                            
                    if valid_shape == True:
                        hasher = hashlib.md5()
                        with open(img, 'rb') as hash_obj:
                            hasher.update(hash_obj.read())
                        hash_value = hasher.hexdigest()

                        if hash_value not in seen_hashes:
                            seen_hashes.add(hash_value)

                            # finally after passing all checks we copy the file to different folder
                            shutil.copy(img, output_dir+ "/" + f"{number_of_conversions:{formatter}}.jpg")
                            number_of_conversions += 1
                        
                        else: 
                            f.write(f"Image: {os.path.basename(os.path.normpath(img))} is a duplicate !! \n" )

                    else: 
                        f.write(f"Image: {os.path.basename(os.path.normpath(img))} is either not rgb or has wrong shape \n" )
            
                else:
                    print(f"Oho! {os.path.basename(os.path.normpath(img))} is too chunky!!")
                    f.write(f"Image: {os.path.basename(os.path.normpath(img))} is too chunky!! \n" )

            else:
                print(f"careful!! some image has wrong ending!!")
                f.write(f"Image: {os.path.basename(os.path.normpath(img))} has no correct ending \n" )

    return number_of_conversions



if __name__ == "__main__":
    base = os.path.dirname(__file__)
    inputs_path = os.path.join(base, "input_images")
    outputs_path = os.path.join(base, "outputs")
    logfile_path = os.path.join(base, "logged_info.log")


    res = validate_images(input_dir=inputs_path, output_dir=outputs_path, log_file=logfile_path)
    print(f"the number is of conversions is {res}")