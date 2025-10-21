"""Base classes for concept models."""

from abc import ABC, abstractmethod
import torch.nn as nn

class ConceptModel(nn.Module, ABC):
    """Minimal unified interface for concept models.

    Implementations should provide:
    - forward(x): return model outputs/logits
    - train(**kwargs): run a training procedure (kwargs include dataset paths, iterations, etc.)
    - interpret(x): return an interpretable explanation object/dict
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""

    @abstractmethod
    def train(self, mode: bool = True, **kwargs):
        """Train / set training mode.

        This method is intentionally compatible with `torch.nn.Module.train(mode)` so
        that calling `model.train(False)` / `model.eval()` continues to work. When
        called with additional kwargs (e.g., dataset/base_network/experiment_name)
        implementations may interpret that as a request to run a training procedure.
        """

    @abstractmethod
    def interpret(self, x):
        """Produce an interpretable explanation for input x."""