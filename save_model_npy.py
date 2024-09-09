import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

import re

import os

# load the model

load_model_dir = './saved_models/vgg19/001/pruned_prototypes_epoch19_k6_pt3'
load_model_name = '19_19_49prune0.7480.pth'

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)

ppnet = torch.load(load_model_path, map_location=torch.device('cpu')) # load model to CPU


for name, param in ppnet.named_parameters():
    # Convert the parameter tensor to numpy array
    param_data = param.detach().cpu().numpy()
    
    # Define the file path
    file_path = f'{name.replace(".", "_")}.npy'
    
    file_save_path = os.path.join('./model_npy/', file_path)

    # Save the numpy array to a file
    np.save(file_save_path, param_data)
    print(f'Saved {name} to {file_path}')