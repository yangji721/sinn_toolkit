"""
Model factory for AWA2 experiments.
"""
import torch.nn as nn

from bcos.models import resnet, densenet, vgg


def get_model(config):
    """
    Get a model for AWA2 experiments.
    
    Parameters
    ----------
    config : dict
        The model configuration.
        
    Returns
    -------
    torch.nn.Module
        The model.
    """
    model_name = config["name"]
    model_args = config.get("args", {})
    bcos_args = config.get("bcos_args", {})
    
    # Get the base model based on name
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            model = resnet.resnet18(**model_args)
        elif model_name == "resnet34":
            model = resnet.resnet34(**model_args)
        elif model_name == "resnet50":
            model = resnet.resnet50(**model_args)
        elif model_name == "resnet101":
            model = resnet.resnet101(**model_args)
        elif model_name == "resnet152":
            model = resnet.resnet152(**model_args)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
    elif model_name.startswith("densenet"):
        if model_name == "densenet121":
            model = densenet.densenet121(**model_args)
        elif model_name == "densenet169":
            model = densenet.densenet169(**model_args)
        elif model_name == "densenet201":
            model = densenet.densenet201(**model_args)
        else:
            raise ValueError(f"Unknown DenseNet model: {model_name}")
    elif model_name.startswith("vgg"):
        if model_name == "vgg11_bnu":
            model = vgg.vgg11_bnu(**model_args)
        else:
            raise ValueError(f"Unknown VGG model: {model_name}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Apply B-cos modifications if needed
    if config.get("is_bcos", False):
        # The model should already be B-cos compatible
        pass
    
    return model
