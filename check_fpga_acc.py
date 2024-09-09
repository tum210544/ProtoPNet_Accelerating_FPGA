import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy
import glob

from helpers import makedir, find_high_activation_crop
from preprocess import mean, std, undo_preprocess_input_function

import torch.nn.functional as F

import cpuinfo


# load the model
load_model_dir = './saved_models/vgg19/004/pruned_prototypes_epoch19_k6_pt3'
load_model_name = '19_19_49prune0.7480.pth'

model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])


load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)

# Check for Intel CPU
cpu_info = cpuinfo.get_cpu_info()
print(f"CPU Brand: {cpu_info['brand_raw']}")
print(f"CPU Vendor: {cpu_info['vendor_id_raw']}")
if 'Intel' in cpu_info['vendor_id_raw']:
    print("Intel CPU is being used.")
else:
    print("Intel CPU is not being used.")

print('load model from ' + load_model_path)
print('model base architecture: ' + model_base_architecture)
print('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path, map_location=torch.device('cpu')) # load model to CPU
ppnet = ppnet.to('cpu')

img_size = ppnet.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

##### SANITY CHECK #####
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

print('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
print('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    print('All prototypes connect most strongly to their respective classes.')
else:
    print('WARNING: Not all prototypes connect most strongly to their respective classes.')

#########################



test_image_conv_dir = './test_images_conv'

image_set_size = 0
correct_predicted = 0

for subdir in os.listdir(test_image_conv_dir):
    subdir_path = os.path.join(test_image_conv_dir, subdir)
    
    # Ensure that it is a directory
    if os.path.isdir(subdir_path):
        image_class = int(subdir.split('.')[0]) - 1
        # Loop through each image file in the subdirectory
        for image_conv_file in glob.glob(os.path.join(subdir_path, '*.npy')):
            image_conv_file_name = os.path.basename(image_conv_file)
            image_conv_file_name = image_conv_file_name[:-11] + ".npy"
            
            image_set_size += 1
            conv_features = torch.tensor(np.load(image_conv_file).reshape(1, 128, 7, 7))
            
            distances = ppnet._l2_convolution(conv_features)

            min_distances = -F.max_pool2d(-distances,
                                kernel_size=(distances.size()[2],
                                            distances.size()[3]))
            min_distances = min_distances.view(-1, ppnet.num_prototypes)
            prototype_activations = ppnet.distance_2_similarity(min_distances)
            logits = ppnet.last_layer(prototype_activations)

            prototype_activation_patterns = ppnet.distance_2_similarity(distances)

            predicted_class = torch.argmax(logits, dim=1)[0].numpy()

            if predicted_class == image_class:
                correct_predicted += 1


print('The image set size: ' + str(image_set_size))
print('\taccu: \t\t{0}%'.format(correct_predicted / image_set_size * 100))