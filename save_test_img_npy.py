import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image


import os
from preprocess import mean, std, undo_preprocess_input_function

import cpuinfo
import glob
from helpers import makedir

# Check for Intel CPU
cpu_info = cpuinfo.get_cpu_info()
print(f"CPU Brand: {cpu_info['brand_raw']}")
print(f"CPU Vendor: {cpu_info['vendor_id_raw']}")
if 'Intel' in cpu_info['vendor_id_raw']:
    print("Intel CPU is being used.")
else:
    print("Intel CPU is not being used.")


img_size = 224
normalize = transforms.Normalize(mean=mean,
                                 std=std)



preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

test_image_dir = './datasets/cub200_cropped/test_cropped'
save_npy_dir = './test_images_npy'
save_conv_dir = './test_images_conv'
makedir(save_npy_dir)
makedir(save_conv_dir)

image_set_size = 0
for subdir in os.listdir(test_image_dir):
    subdir_path = os.path.join(test_image_dir, subdir)
    
    # Ensure that it is a directory
    if os.path.isdir(subdir_path):
        # Loop through each image file in the subdirectory
        save_analysis_folder_path = os.path.join(save_npy_dir, subdir)
        makedir(save_analysis_folder_path)
        save_conv_folder_dir = os.path.join(save_conv_dir, subdir)
        makedir(save_conv_folder_dir)
        for image_file in glob.glob(os.path.join(subdir_path, '*.jpg')):
            image_file_name = os.path.splitext(os.path.basename(image_file))[0]
            save_analysis_path = os.path.join(save_analysis_folder_path, image_file_name)
            image_set_size += 1
            img_pil = Image.open(image_file)

            # Convert to RGB if the image is grayscale
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert("RGB")
            img_tensor = preprocess(img_pil) 
            images_test = Variable(img_tensor.unsqueeze(0))

            image_test_squeeze = np.squeeze(images_test.numpy(), axis=0)
            np.save(save_analysis_path + '.npy', image_test_squeeze)

print(image_set_size)