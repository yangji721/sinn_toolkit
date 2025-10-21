import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from concept_models.tesnet.model_cap import construct_TesNet
from concept_models.tesnet.train_and_test_cap import train as train_model
from concept_models.tesnet.train_and_test_cap import test as test_model
from concept_models.tesnet.train_and_test_cap import joint, warm_only

class TesNetAdapter:
    """Adapter for TesNet model implementing standard concept model interface."""
    
    def __init__(
        self,
        num_classes: int = 200,
        num_prototypes: int = 2000,
        feature_dimension: int = 128,
        prototype_activation_function: str = 'log',
        cap_width: float = 6.0,
        img_size: int = 224,
        base_architecture: str = 'resnet50',
        device: Optional[torch.device] = None,
        **kwargs
    ):
        """Initialize TesNet model adapter.
        
        Args:
            num_classes: Number of output classes
            num_prototypes: Number of prototypes to use (must be divisible by num_classes)
            feature_dimension: Dimension of feature space for prototypes
            prototype_activation_function: Activation function for prototype layer ('log' or 'linear')
            cap_width: Width parameter for capsule units
            img_size: Size of input images 
            base_architecture: Base CNN architecture to use ('resnet50', 'resnet18', etc)
            device: PyTorch device to use
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.device = device
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.feature_dimension = feature_dimension
        self.img_size = img_size
        self.cap_width = cap_width
        
        # Create model
        prototype_shape = (num_prototypes, feature_dimension, 1, 1)
        self.model = construct_TesNet(
            base_architecture=base_architecture,
            pretrained=True,
            img_size=img_size,
            prototype_shape=prototype_shape,
            num_classes=num_classes,
            cap_width=cap_width,
            prototype_activation_function=prototype_activation_function
        )
        self.model = self.model.to(device)
        self.model = torch.nn.DataParallel(self.model)
        
        # Training configurations
        self.optimizer = None
        self.joint_optimizer = None
        self.last_layer_optimizer = None
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Set up the various optimizers used in training."""
        # Joint optimizer for all parts
        joint_optimizer_specs = [
            {'params': self.model.module.features.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
            {'params': self.model.module.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
            {'params': self.model.module.prototype_vectors, 'lr': 3e-3},
            {'params': self.model.module.cap_width_l2, 'lr': 3e-3}
        ]
        self.joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

        # Warm optimizer for non-features parts
        warm_optimizer_specs = [
            {'params': self.model.module.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
            {'params': self.model.module.prototype_vectors, 'lr': 3e-3},
            {'params': self.model.module.cap_width_l2, 'lr': 3e-3}
        ]
        self.warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        # Last layer optimizer
        last_layer_specs = [{'params': self.model.module.last_layer.parameters(), 'lr': 1e-4}]
        self.last_layer_optimizer = torch.optim.Adam(last_layer_specs)

    def train(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 100,
        warm_epochs: int = 5,
        coefs: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Train the TesNet model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Total number of epochs to train
            warm_epochs: Number of warmup epochs where only non-feature parts are trained
            coefs: Dict of loss coefficients
            
        Returns:
            Dict containing training metrics
        """
        if coefs is None:
            coefs = {
                'crs_ent': 1.0,  # Cross entropy loss
                'clst': 0.8,     # Clustering loss
                'sep': 0.08,     # Separation loss
                'l1': 1e-4,      # L1 regularization
                'orth': 1.0,     # Orthogonality loss
                'sub_sep': 1e-7, # Subspace separation loss
                'cap_coef': 3e-3 # Capsule loss coefficient
            }
            
        best_acc = 0.0
        metrics = {}
        
        for epoch in range(epochs):
            # Warm epochs - train only non-feature parts
            if epoch < warm_epochs:
                warm_only(model=self.model)
                train_acc, train_metrics = train_model(
                    model=self.model,
                    dataloader=train_loader,
                    optimizer=self.warm_optimizer,
                    class_specific=True,
                    coefs=coefs
                )
            else:
                # Joint training of all parts
                joint(model=self.model)
                train_acc, train_metrics = train_model(
                    model=self.model,
                    dataloader=train_loader,
                    optimizer=self.joint_optimizer,
                    class_specific=True,
                    coefs=coefs
                )
                
            # Validation
            if val_loader is not None:
                val_acc, val_metrics = test_model(
                    model=self.model,
                    dataloader=val_loader,
                    class_specific=True
                )
                best_acc = max(best_acc, val_acc)
                
            metrics = {
                'train_acc': train_acc,
                'best_val_acc': best_acc,
                **train_metrics
            }
                
        return metrics
    
    def interpret(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get interpretable components from the model for given input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dict containing interpretable components:
                - 'logits': Class logits
                - 'prototype_activations': Prototype activation values
                - 'prototype_similarities': Cosine similarities to prototypes
                - 'concept_cap_widths': Capsule width parameters
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass 
            x = x.to(self.device)
            logits, cos_distances, proto_acts, cap_factors = self.model(x)
            
            interpretations = {
                'logits': logits,
                'prototype_activations': proto_acts,
                'prototype_similarities': -cos_distances,  # Convert distance to similarity
                'concept_cap_widths': cap_factors
            }
            
            return interpretations
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits.
        
        Args:
            x: Input tensor
            
        Returns:
            Class logits tensor
        """
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits, _, _, _ = self.model(x)
            return logits

    def save(self, path: Union[str, Path]):
        """Save model state to disk.
        
        Args:
            path: Path to save model state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: Union[str, Path]):
        """Load model state from disk.
        
        Args:
            path: Path to load model state from
        """
        self.model.load_state_dict(torch.load(path))
        
    @property
    def prototype_vectors(self) -> torch.Tensor:
        """Get the prototype vectors learned by the model."""
        return self.model.module.prototype_vectors.data
        
    @property 
    def capsule_widths(self) -> torch.Tensor:
        """Get the learned capsule width parameters."""
        return self.model.module.cap_width_l2.data