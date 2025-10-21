"""High-level convenience API wrapping original bcos factories.

Provides:
- create_model(name, pretrained=True, **kwargs): builds a B-cos model by name.
- list_pretrained(): lists available pretrained model entrypoints.
- train(args): thin wrapper to call the original trainer.run_training with an argparse-like namespace.
"""
from types import SimpleNamespace
from typing import Any, Dict, List

import bcos.models.pretrained as pretrained
from bcos.models import pretrained as _pretrained_module
from bcos.experiments.utils import get_configs_and_model_factory
from bcos.training import trainer as _trainer
from concept_models.bcos.model import BcosModelAdapter


def list_pretrained() -> List[str]:
    """List available pretrained entrypoints from original B-cos package."""
    if hasattr(pretrained, "list_available"):
        return pretrained.list_available()
    # fallback: inspect module attributes
    return [n for n in dir(_pretrained_module) if not n.startswith("_")]


def create_model(name: str, pretrained_weights: bool = True, **kwargs: Any):
    """Create a B-cos model by name.

    This will try the following, in order:
    1. Look for a pretrained entrypoint with that name (e.g., 'resnet50') and call it.
    2. Try to use the experiments-based factory if kwargs contain 'dataset' and 'base_network' and an 'experiment_name'.

    Parameters
    ----------
    name: str
        Name of the model entrypoint (e.g., 'resnet50').
    pretrained_weights: bool
        Whether to load pretrained weights via the pretrained entrypoint.
    **kwargs:
        Extra kwargs forwarded to the model constructors or experiment factory.
    """
    # 1) try pretrained registry
    if hasattr(pretrained, name):
        # return adapter wrapping the original model so it implements unified interface
        return BcosModelAdapter(model_name=name, pretrained=pretrained_weights, **kwargs)

    # 2) try experiments factory if provided
    dataset = kwargs.get("dataset")
    base_network = kwargs.get("base_network")
    experiment_name = kwargs.get("experiment_name", name)
    if dataset and base_network:
        # use experiments utils to get factory
        configs, model_factory = get_configs_and_model_factory(dataset, base_network)
        # if experiment_name not in configs, raise
        if experiment_name not in configs:
            raise RuntimeError(f"Unknown experiment '{experiment_name}' for {base_network}/{dataset}")
        # model factory follows signature get_model(model_config)
        model_config = configs[experiment_name]
        return model_factory(model_config)

    raise RuntimeError(f"Could not create model '{name}': not found in pretrained registry and no experiment info provided.")


def train(args: Dict[str, Any]):
    """Thin wrapper to call original bcos training.

    Args can be a dict or a SimpleNamespace. This function will convert
    it into a namespace with expected attributes and call trainer.run_training().
    """
    if isinstance(args, dict):
        args = SimpleNamespace(**args)
    # Ensure required attributes exist with sane defaults
    defaults = dict(
        base_directory="./experiments",
        dataset="ImageNet",
        base_network="bcos_final",
        experiment_name="resnet_50",
        resume=True,
        jit=False,
        csv_logger=False,
        tensorboard_logger=False,
        wandb_logger=False,
        wandb_project=None,
        wandb_id=None,
        wandb_name=None,
        fast_dev_run=False,
        distributed=False,
        cache_dataset=None,
        explanation_logging=False,
        explanation_logging_every_n_epochs=1,
        nodes=1,
        track_grad_norm=False,
        amp=False,
        debug=False,
        refresh_rate=None,
        wandb_name_override=None,
        base_network_override=None,
    )

    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Call the original trainer
    return _trainer.run_training(args)
