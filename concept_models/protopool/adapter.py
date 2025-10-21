import torch
import torch.nn as nn
from ..base import ConceptModel

# Import Protopool_cap lazily inside constructor to avoid import-time errors
from .settings import img_size, prototype_shape, num_classes


class ProtoPoolAdapter(ConceptModel):
    """Adapter wrapping ProtoPool (Protopool_cap) to the unified ConceptModel interface."""

    def __init__(self, base_model=None, img_size=img_size, prototype_shape=prototype_shape, num_classes=num_classes):
        super().__init__()
        if base_model is None:
            # instantiate default Protopool_cap using repo defaults
            # import here to avoid module resolution issues at package import time
            from .model_cap_final import Protopool_cap
            # match Protopool_cap constructor signature (num_prototypes, num_descriptive, num_classes,...)
            # use prototype_shape[0] as num_prototypes and set num_descriptive=1 by default
            num_prototypes = prototype_shape[0]
            num_descriptive = 1
            proto_depth = prototype_shape[1] if len(prototype_shape) > 1 else 128
            # use a supported add_on_layers_type ('regular') to avoid NotImplementedError
            self.model = Protopool_cap(num_prototypes, num_descriptive, num_classes,
                                      arch='resnet50', pretrained=True,
                                      add_on_layers_type='regular', proto_depth=proto_depth)
        else:
            self.model = base_model

        self.prototype_shape = prototype_shape
        self.num_classes = num_classes

    def forward(self, x):
        # Protopool may return logits only or (logits, extra)
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out

    def train(self, mode: bool = True, **kwargs):
        # Same pattern as other adapters: if kwargs empty behave like nn.Module.train()
        if not kwargs:
            self.training = mode
            self.model.train(mode)
            return self

        # training runner is left to user; we provide a simple loop if requested
        train_loader = kwargs.get('train_loader')
        if train_loader is None:
            raise ValueError('train_loader required to run full training')
        epochs = kwargs.get('epochs', 10)
        optimizer = kwargs.get('optimizer', torch.optim.Adam(self.model.parameters()))

        for epoch in range(epochs):
            self.model.train()
            for data, target in train_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                logits = self.forward(data)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()

    def interpret(self, x):
        """Return interpretability artifacts from Protopool model (prototype activations, distances, weights)
        We'll attempt to call common attributes found in the Protopool implementation and gracefully
        fall back to basic outputs if missing.
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            # try to find prototype activations and distances in returned object or model attributes
            proto_act = None
            distances = None
            if isinstance(out, tuple):
                # common pattern: (logits, distances, ...)
                if len(out) >= 2:
                    distances = out[1]
                if len(out) >= 3:
                    proto_act = out[2]

            # fallback: try model attributes
            if proto_act is None and hasattr(self.model, 'push_forward'):
                proto_act = self.model.push_forward(x)

            if distances is None and hasattr(self.model, 'prototype_distances'):
                distances = self.model.prototype_distances(x)

            logits = out[0] if isinstance(out, tuple) else out
            class_scores = torch.softmax(logits, dim=1)

            return {
                'logits': logits,
                'class_scores': class_scores,
                'prototype_activations': proto_act,
                'prototype_distances': distances,
            }
