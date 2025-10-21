"""ProtoPNet model implementation with unified interface."""

from .adapter import ProtoPNetModelAdapter
from .features import *
from .model import PPNet
from .settings import base_architecture, img_size, prototype_shape, num_classes

__all__ = [
    'ProtoPNetModelAdapter',
    'PPNet',
    'base_architecture',
    'img_size',
    'prototype_shape',
    'num_classes'
]