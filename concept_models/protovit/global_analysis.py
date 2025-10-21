import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2
import matplotlib.pyplot as plt

import re

import os

from helpers import makedir
import find_nearest
import argparse

from preprocess import preprocess_input_function

# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
#parser.add_argument('-modeldir', nargs=1, type=str)
#parser.add_argument('-model', nargs=1, type=str)
args = parser.parse_args()

from analysis_settings import load_model_dir, load_model_name, img_name
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = load_model_dir
load_model_name = load_model_name

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

# load the model
print('Load model from ' + load_model_path)
print('start_epoch_number: ', start_epoch_number)
ppnet = torch.load(load_model_path)
ppnet = ppnet.cuda()
#ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = ppnet.img_size

# load the data
# must use unaugmented (original) dataset
from settings import train_push_dir, test_dir,train_push_batch_size,test_batch_size

train_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_push_batch_size, shuffle=True,
    num_workers=2, pin_memory=False)


# test set: do not normalize
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True,
    num_workers=2, pin_memory=False)

root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test')

makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)


def save_prototype(fname,img_dir):
    p_img = plt.imread(img_dir)
    plt.imsave(fname, p_img)

    
prototype_img_filename_prefix='prototype-img'
for j in range(ppnet.num_prototypes):
    makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
    makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
    load_img_dir = os.path.join(load_model_dir, img_name)

    bb_dir = os.path.join(load_img_dir, prototype_img_filename_prefix + 'bbox-original' + str(j) +'.png')
    saved_bb_dir_tr =os.path.join(root_dir_for_saving_train_images, str(j),
                                                             'prototype_in_original_bb.png')
    # save for training imgs
    save_prototype(saved_bb_dir_tr,bb_dir)
    # save for test imgs
    saved_bb_dir_ts =os.path.join(root_dir_for_saving_test_images, str(j),
                                                             'prototype_in_original_bb.png')
    save_prototype(saved_bb_dir_ts,bb_dir)

num_nearest_neighbors = 5
find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        ppnet=ppnet, # pytorch network with prototype_vectors
        num_nearest_neighbors=num_nearest_neighbors,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print)

find_nearest.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        ppnet=ppnet, # pytorch network with prototype_vectors
        num_nearest_neighbors=num_nearest_neighbors,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print)