"""
General settings. Mainly paths to data.
"""
import os

# data root (mainly for CIFAR10)
DATA_ROOT = os.getenv("DATA_ROOT")

# ImageNet path
IMAGENET_PATH = os.getenv("IMAGENET_PATH")

# CUB dataset path
CUB_PATH = os.getenv("CUB_PATH", "/data/yang/benchmark/data/CUB_processed")

# AWA2 dataset path
AWA2_PATH = os.getenv("AWA2_PATH", "/data/yang/benchmark/data/AWA2_processed")

# DeepFashion dataset path
DEEPFASHION_PATH = os.getenv("DEEPFASHION_PATH", "/data/yang/benchmark/data/DeepFashion_processed")

# ---------------------------------------------
# Following are only needed for caching!!!
# ---------------------------------------------
SHMTMPDIR = "/dev/shm"

# ImageNet train data tar files
# Note this is not the same as the official ImageNet tar file
# this expects each class to inside its own tar file, which will then be extracted
# to SHMTMPDIR
IMAGENET_TRAIN_TAR_FILES_DIR = os.getenv("IMAGENET_TRAIN_TAR_FILES_DIR")

# Path to redis server
REDIS_SERVER = os.getenv("REDIS_SERVER_PATH")
