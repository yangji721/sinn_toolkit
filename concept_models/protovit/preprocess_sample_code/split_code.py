import torch 
import os 
import matplotlib.pyplot as plt 
import shutil
from tqdm import tqdm
from PIL import Image 

data_path = r"/data/yang/benchmark/data/CUB"
index_loc = os.path.join(data_path, "images.txt")
index_dict = open(index_loc, 'r')
index = {}

img_loc = os.path.join(data_path,  "images")

for line in index_dict.readlines():
    word = line.split()
    key = int(word[0])
    image_address = os.path.join(img_loc, word[1])
    index[key] = image_address

label_loc = os.path.join(data_path, "image_class_labels.txt")
label_dict = {}
for line in open(label_loc, 'r').readlines():
    word = line.split()
    key = int(word[0])
    label_dict[key] = int(word[1])

class_ref = {}
ref_loc = os.path.join(data_path, "classes.txt")
for line in open(ref_loc, 'r').readlines():
    word = line.split()
    key = int(word[0])
    class_ref[key] = word[1]

split_loc = os.path.join(data_path, "train_test_split.txt")
train_dict = {}
for line in open(split_loc, 'r').readlines():
    word = line.split()
    key = int(word[0])
    train_dict[key] = int(word[1])

for i in tqdm(index):
    root = index[i]
    img = Image.open(root)
    name = index[i].split('/')[-1]
    label = label_dict[i]
    train = train_dict[i]
    reference = class_ref[label].replace('.','_')
    if train ==0:
        file_name =r"/data/yang/benchmark/data/CUB_2011_200_uncropped/train"
    elif train ==1:
        file_name = r"/data/yang/benchmark/data/CUB_2011_200_uncropped/test"
    doc_root = os.path.join(file_name,reference)
    if not os.path.exists(doc_root):
        os.makedirs(doc_root)
    image_root = os.path.join(doc_root, name)
    img.save(image_root)

