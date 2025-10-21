"""B-cos model adapter implementing the common concept model interface."""

import torch
import torch.nn as nn
from concept_models.base import ConceptModel
import bcos.models as models
from bcos.common import explanation_mode, gradient_to_image
from bcos.data.presets import ImageNetClassificationPresetEval


class BcosModelAdapter(ConceptModel):
    """Adapter for B-cos models implementing the common concept model interface."""
    
    def __init__(self, model_name="resnet50", pretrained=True, **kwargs):
        """Initialize B-cos model.
        
        Args:
            model_name (str): Name of the B-cos model architecture (e.g., resnet50, densenet121)
            pretrained (bool): Whether to load pretrained weights
            **kwargs: Additional arguments passed to the model constructor
        """
        super().__init__()
        # Get the model builder function
        model_fn = getattr(models.pretrained, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model {model_name} not found in B-cos models")
            
        # Create the model
        self.model = model_fn(pretrained=pretrained, **kwargs)
        self.model_name = model_name
        # expose original model utilities if present
        self._orig_model = self.model
        # attach transform (pretrained entrypoints usually attach transform)
        self.transform = getattr(self.model, "transform", None)
        # fallback: use a preset that applies AddInverse for B-cos inputs
        if self.transform is None:
            # default crop_size 224
            self.transform = ImageNetClassificationPresetEval(crop_size=224, is_bcos=True)
        
    def forward(self, x):
        """Forward pass returning class logits."""
        return self.model(x)
    
    def get_concepts(self, x):
        """Get concept activations for the input.
        
        In B-cos networks, the concept information is embedded in the alignment
        between feature vectors and weight vectors. We return the alignment scores.
        """
        # Enable explanation mode to get alignments
        ctx = None
        if hasattr(self._orig_model, "explanation_mode"):
            ctx = self._orig_model.explanation_mode()
        else:
            ctx = explanation_mode(self._orig_model)

        with ctx:
            output = self.model(x)
            # Get alignment scores from the last layer
            # This varies by model architecture but generally available through model hooks
            if hasattr(self.model, "get_last_alignment"):
                alignments = self.model.get_last_alignment()
                return {"concept_scores": alignments}
            # fallback: try BcosUtilMixin.explain to obtain dynamic linear weights
            if hasattr(self.model, "explain"):
                # explain expects batch size 1
                try:
                    expl = self.model.explain(x)
                    return {
                        "dynamic_linear_weights": expl.get("dynamic_linear_weights"),
                        "contribution_map": expl.get("contribution_map"),
                        "explanation": expl.get("explanation"),
                        "prediction": expl.get("prediction"),
                    }
                except Exception:
                    return {"concept_scores": None}
            return {"concept_scores": None}
    
    def interpret(self, x):
        """Get interpretable representation showing concept contributions.
        
        Returns:
            dict: Dictionary containing:
                - 'contribution_map': Tensor showing which image regions contributed to the prediction
                - 'concept_scores': Alignment scores between features and weight vectors
                - 'prediction': Model's prediction
        """
        # Enable explanation mode
        if hasattr(self._orig_model, "explanation_mode"):
            ctx = self._orig_model.explanation_mode()
        else:
            ctx = explanation_mode(self._orig_model)

        with ctx:
            output = self.model(x)
            pred = output.argmax(dim=1)

            # Get gradient-based contribution map
            x.requires_grad_(True)
            output[:, pred].sum().backward()
            # gradient_to_image expects (image, linear_mapping, ...)
            # x shape: (B, C, H, W) and x.grad same; use first element
            contrib_map = gradient_to_image(x[0], x.grad[0])

            # Get concept alignments
            if hasattr(self.model, "get_last_alignment"):
                alignments = self.model.get_last_alignment()
            else:
                alignments = None

        # If we didn't obtain alignments above, try the explain helper
        if alignments is None and hasattr(self.model, "explain"):
            try:
                expl = self.model.explain(x)
                # prefer explain output but include our contrib_map as well
                result = dict(
                    prediction=expl.get("prediction", pred),
                    contribution_map=expl.get("contribution_map", contrib_map),
                    concept_scores=expl.get("dynamic_linear_weights", alignments),
                    explanation=expl.get("explanation"),
                )
                return result
            except Exception:
                pass

        return {
            "contribution_map": contrib_map,
            "concept_scores": alignments,
            "prediction": pred,
        }

    def train(self, mode: bool = True, **kwargs):
        """Either set training mode or run full training when kwargs supplied.

        - If only called as `train(True/False)`, behave like nn.Module.train(mode).
        - If called with dataset/base_network/experiment_name kwargs, delegate to trainer.run_training.
        """
        # If called as mode toggle (typical torch usage), just set attribute
        if not kwargs:
            # mirror torch.nn.Module.train behavior
            self.training = bool(mode)
            return self

        # If experiment info is provided, delegate to bcos training.trainer.run_training
        if "dataset" in kwargs and "base_network" in kwargs and "experiment_name" in kwargs:
            # build a simple namespace expected by trainer.run_training
            from types import SimpleNamespace
            args_ns = SimpleNamespace(**kwargs)
            # ensure some defaults
            for k, v in dict(resume=False, fast_dev_run=False, base_directory=kwargs.get("base_directory", "./experiments")).items():
                if not hasattr(args_ns, k):
                    setattr(args_ns, k, v)
            from bcos.training import trainer as _trainer
            return _trainer.run_training(args_ns)

        raise NotImplementedError("Training must be invoked with dataset, base_network and experiment_name or implemented per-model.")