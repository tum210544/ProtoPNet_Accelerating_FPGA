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
from log import create_logger
from preprocess import mean, std, undo_preprocess_input_function

import time
import torch.nn.functional as F

import cpuinfo

# Set PyTorch to use only one thread (single-core)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

start = time.time()

test_image_dir = './datasets/cub200_cropped/test_cropped'

# load the model
load_model_dir = './saved_models/vgg19/001/pruned_prototypes_epoch19_k6_pt3'
load_model_name = '19_19_49prune0.7480.pth'

model_base_architecture = load_model_dir.split('/')[2]
experiment_run = '/'.join(load_model_dir.split('/')[3:])

save_analysis_folder_path = os.path.join('./execution_analysis', model_base_architecture,
                                  '001', load_model_name)
makedir(save_analysis_folder_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_folder_path, 'execution_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

# Check for Intel CPU
cpu_info = cpuinfo.get_cpu_info()
log(f"CPU Brand: {cpu_info['brand_raw']}")
log(f"CPU Vendor: {cpu_info['vendor_id_raw']}")
if 'Intel' in cpu_info['vendor_id_raw']:
    log("Intel CPU is being used.")
else:
    log("Intel CPU is not being used.")

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

ppnet = torch.load(load_model_path, map_location=torch.device('cpu')) # load model to CPU
ppnet = ppnet.to('cpu')

img_size = ppnet.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

normalize = transforms.Normalize(mean=mean,
                                 std=std)
 
##### SANITY CHECK #######
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

######################


preprocess = transforms.Compose([
   transforms.Resize((img_size,img_size)),
   transforms.ToTensor(),
   normalize
])

end = time.time()

start_2 = time.time()

image_set_size = 0
correct_predicted = 0

image_preprocess_duration = 0
convolution_duration = 0
prototype_layer_duration = 0
last_layer_duration = 0

for subdir in os.listdir(test_image_dir):
    subdir_path = os.path.join(test_image_dir, subdir)
    
    # Ensure that it is a directory
    if os.path.isdir(subdir_path):
        image_class = int(subdir.split('.')[0]) - 1
        # Loop through each image file in the subdirectory
        for image_file in glob.glob(os.path.join(subdir_path, '*.jpg')):

            start_loop = time.time()

            image_file_name = os.path.basename(image_file)
            save_analysis_path = os.path.join(save_analysis_folder_path, subdir, image_file_name)
            makedir(save_analysis_path)
            image_set_size += 1
            img_pil = Image.open(image_file)

            # Convert to RGB if the image is grayscale
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert("RGB")
            img_tensor = preprocess(img_pil) 
            img_variable = Variable(img_tensor.unsqueeze(0))
            images_test = img_variable
            images_test = images_test.to('cpu')
            
            end_image_preprocess = time.time()
            image_preprocess_duration += end_image_preprocess - start_loop
            
            conv_features = ppnet.conv_features(images_test)
            end_conv = time.time()
            convolution_duration += end_conv - end_image_preprocess

            distances = ppnet._l2_convolution(conv_features)
            min_distances = -F.max_pool2d(-distances,
                                kernel_size=(distances.size()[2],
                                            distances.size()[3]))
            min_distances = min_distances.view(-1, ppnet.num_prototypes)
            prototype_activations = ppnet.distance_2_similarity(min_distances)
            
            prototype_activation_patterns = ppnet.distance_2_similarity(distances)

            end_prototype = time.time()
            prototype_layer_duration += end_prototype - end_conv

            logits = ppnet.last_layer(prototype_activations)
            predicted_class = torch.argmax(logits, dim=1)[0].numpy()

            if predicted_class == image_class:
                correct_predicted += 1

            end_prediction = time.time()
            last_layer_duration += end_prediction - end_prototype

end_2 = time.time()
total_inference_duration = end_2 -  start_2
log('\ttime before load images: \t{0}'.format(end -  start))
log('The image set size: ' + str(image_set_size))
log('\ttotal inference duration: \t{0}'.format(total_inference_duration))
log('\tavg inference duration: \t{0}'.format(total_inference_duration/image_set_size))
log('\taccu: \t\t{0}%'.format(correct_predicted / image_set_size * 100))

log('\tavg image preprocess duration: \t{0}'.format(image_preprocess_duration/image_set_size))
log('\tavg convolutional layer duration: \t{0}'.format(convolution_duration/image_set_size))
log('\tavg prototype layer duration: \t{0}'.format(prototype_layer_duration/image_set_size))
log('\tavg last layer duration: \t{0}'.format(last_layer_duration/image_set_size))

logclose()

