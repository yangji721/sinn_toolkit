# ProtoViT: Interpretable Image Classification with Adaptive Prototype-based Vision Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper ["**Interpretable Image Classification with Adaptive Prototype-based Vision Transformers**"](https://arxiv.org/abs/2410.20722). **(NeurIPS 2024)**

<div align="center">
<img src="assets/arch.jpg" width="600px">
</div>

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Analysis](#analysis)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

ProtoViT is a novel approach that combines Vision Transformers with prototype-based learning to create interpretable image classification models. Our implementation provides both high accuracy and explainability through learned prototypes.

## Prerequisites

### Software Requirements

These packages should be enough to reproduce our results. We add requirement.txt based on our conda environment for reference just in case. 
- Python 3.8+
- PyTorch with cuda
- NumPy
- OpenCV (cv2)
- [Augmentor](https://github.com/mdbloice/Augmentor)
- Timm==0.4.12 (Note: Higher versions may require modifications to the ViT encoder)

### Hardware Requirements
Recommended GPU configurations:
- 1× NVIDIA Quadro RTX 6000 (24GB) or
- 1× NVIDIA GeForce RTX 4090 (24GB) or
- 1× NVIDIA RTX A6000 (48GB)

## Installation

```bash
git clone https://github.com/Henrymachiyu/ProtoViT.git
cd ProtoViT
pip install -r requirements.txt
```

## Dataset Preparation

### CUB-200-2011 Dataset

1. Download [CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Extract the dataset:
   ```bash
   #Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
   tar -xzf CUB_200_2011.tgz
   ```
3. Process the dataset:
   
   For cropping data and training_test split, please carefully follow the instructions from the dataset.
   Sample code can be found in preprocess sample code that can crop and split data with Jupyter Notebook.
      
   ```bash
   # Create directory structure
   mkdir -p ./datasets/cub200_cropped/{train_cropped,test_cropped}
   
   # Crop and split images using provided scripts
   python your_own_scripts/crop_images.py  # Uses bounding_boxes.txt
   python your_own_scripts/split_dataset.py  # Uses train_test_split.txt
   #Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
   #Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
   
   # Augment training data 
   python img_aug.py
   #this will create an augmented training set in the following directory:
   #"./datasets/cub200_cropped/train_cropped_augmented/"
   ```

### Stanford Cars Dataset
The official website for the dataset is: 
- [Official Stanford Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

Alternative dataset option available from:
- [Kaggle Mirror](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/data)

## Training

1. Configure settings in `settings.py`:

```python
# Dataset paths
data_path = "./datasets/cub200_cropped/"
train_dir = data_path + "train_cropped_augmented/"
test_dir = data_path + "test_cropped/"
train_push_dir = data_path + "train_cropped/"
```

2. Start training:
```bash
python main.py
```

## Analysis

### Parameter settings 

The corresponsing parameter settings for global and local analysis are saved in the analysis_settings.py 
```python 
   load_model_dir = 'saved model path'#'./saved_models/vgg19/003/'
   load_model_name = 'model_name'#'14finetuned0.9230.pth'
   save_analysis_path = 'saved_dir_rt'
   img_name = 'prototype_vis_file'# 'img/'
   test_data = "test_dir"
   check_test_acc = False
   check_list =['list of test images'] #"163_Mercedes-Benz SL-Class Coupe 2009/03123.jpg", Could be a list of images
```

### Local Analysis and reasoning process

To produce the reasoning plots: 
<div align="center">
<img src="assets/reasoning.jpg" width="600px">
</div>

We analyze nearest prototypes for specific test images and retrieve model reasoning process for predictions:

```bash
# this function provdes results for model's reasoning and local analysis

python local_analysis.py -gpuid 0
```

### Global Analysis
To produce the global analysis plots:

<div align="center">
<img src="assets/analysis.jpg" width="600px">
</div>

This following file finds nearest patches for each prototype to ensure the prototypes are semantically consistent across samples in train and test data:

```bash
python global_analysis.py -gpuid 0
```

## Location Misalignment 

To run the experiment, you would also need cleverhans
```bash
pip install cleverhans
```
### Parameter settings 

All the parameters used for reproducing our results on location misalignment are stored in adv_settings.py 

```python
load_model_path = "."
test_dir = "./cub200_cropped/test_cropped"
model_output_dir = "." # dir for saving all the results 
```

To run the adversarial attack and retrieve the results

```bash
cd ./spatial_alignment_test
python run_adv_test.py # as default, we ran experiment over entire test set
```

### Model Zoo

### CUB-200-2011 Dataset Results
We provide checkpoints after projection and last layer finetuning on CUB-200-2011 dataset. 
| Model Version | Backbone | Resolution | Accuracy | Checkpoint |
|--------------|----------|------------|----------|------------|
| ProtoViT-T | DeiT-Tiny | 224×224 | 83.36% | [Download](https://huggingface.co/chiyum609/ProtoViT/blob/main/DeiT_Tiny_finetuned0.8336.pth) | 
| ProtoViT-S | DeiT-Small | 224×224 | 85.30% | [Download](https://huggingface.co/chiyum609/ProtoViT/blob/main/DeiT_Small_finetuned0.8530.pth) | 
| ProtoViT-CaiT | CaiT_xxs24 | 224×224 | 86.02% |  [Download](https://huggingface.co/chiyum609/ProtoViT/blob/main/CaiT_xxs24_224_finetuned0.8602.pth) |


## Acknowledgments

This implementation is based on the timm, [ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) repository and its variations. We thank the authors for their valuable work.

## Contact Info

If you have any questions regarding the paper or implementations, please don't hesitate to email us: chiyu.ma.gr@dartmouth.edu

Feel free to ⭐ the repo, contribute, or share it with others who might find it useful!

## Citation 

If you find this work helpful in your research, please also consider citing:

```bibtex 
@inproceedings{ma2024interpretable,
  title={Interpretable Image Classification with Adaptive Prototype-based Vision Transformers},
  author={Ma, Chiyu and Donnelly, Jon and Liu, Wenjun and Vosoughi, Soroush and Rudin, Cynthia and Chen, Chaofan},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
}
```
## License

This project is licensed under the MIT License. 
