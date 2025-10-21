import torch
import torch.nn as nn
from ..base import ConceptModel
from .model import PPNet
from .settings import base_architecture, img_size, prototype_shape, num_classes

class ProtoPNetModelAdapter(ConceptModel):
    """Adapter for ProtoPNet model implementing the unified concept model interface"""
    
    def __init__(self, base_model=None, base_architecture=base_architecture, 
                 num_classes=num_classes, prototype_shape=prototype_shape,
                 pretrained=True, img_size=img_size):
        super().__init__()
        
        if base_model is None:
            # base_architecture is a function that creates the feature extractor
            self.model = PPNet(
                features=base_architecture,
                img_size=img_size, 
                prototype_shape=prototype_shape,
                num_classes=num_classes,
                init_weights=True
            )
            if torch.cuda.is_available():
                self.model = self.model.cuda()
        else:
            self.model = base_model
            
        self.num_classes = num_classes
        self.prototype_shape = prototype_shape
        
    def forward(self, x):
        """Forward pass returning class logits"""
        logits, min_distances = self.model(x)
        return logits
    
    def train(self, mode: bool = True, **kwargs):
        """Set training mode or run training procedure.
        
        Args:
            mode (bool): If no other args provided, sets training mode like nn.Module.train()
            **kwargs: If provided, runs full training procedure with these parameters:
                train_loader: DataLoader for training data
                val_loader: Optional DataLoader for validation
                epochs: Number of epochs (default: 10)
                optimizer: Optional optimizer (default: Adam)
                scheduler: Optional learning rate scheduler
        """
        # If only mode is provided, behave like nn.Module.train()
        if not kwargs:
            self.training = mode
            self.model.train(mode)
            return self
            
        # Otherwise, run full training procedure
        train_loader = kwargs.get('train_loader')
        if train_loader is None:
            raise ValueError("train_loader is required for training")
            
        val_loader = kwargs.get('val_loader')
        epochs = kwargs.get('epochs', 10)
        optimizer = kwargs.get('optimizer', None)
        scheduler = kwargs.get('scheduler', None)
        
        if optimizer is None:
            optimizer = torch.optim.Adam([
                {'params': self.model.features.parameters(), 'lr': 0.0001},
                {'params': self.model.add_on_layers.parameters(), 'lr': 0.003},
                {'params': self.model.prototype_vectors, 'lr': 0.003},
            ])
            
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for i, (images, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Move data to GPU if available
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                
                # Forward pass
                logits, min_distances = self.model(images)
                
                # Compute classification loss
                cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
                
                # Add prototype-specific losses if available
                if hasattr(self.model, 'push_forward'):
                    # Get prototype activations
                    prototype_activations = self.model.push_forward(images)
                    
                    # Add clustering loss and separation loss
                    clustering_loss = self.model.cluster_loss(prototype_activations)
                    separation_loss = self.model.separation_loss(min_distances, labels)
                    
                    # Total loss
                    loss = cross_entropy + 0.8 * clustering_loss + 0.08 * separation_loss
                else:
                    loss = cross_entropy
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            # Print epoch statistics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            
            # Validation if provided
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                print(f'Validation Accuracy: {val_acc:.2f}%')
            
            if scheduler is not None:
                scheduler.step()
                
    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                    
                logits, _ = self.model(images)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        accuracy = 100. * correct / total
        return accuracy
    
    def interpret(self, x):
        """Get interpretable concepts and their contributions"""
        self.model.eval()
        with torch.no_grad():
            # Get prototype similarities and classification logits
            logits, distances = self.model(x)
            
            # Get prototype activations
            if hasattr(self.model, 'push_forward'):
                prototype_activations = self.model.push_forward(x)
            else:
                prototype_activations = -distances  # Higher value means more similar
            
            # Get final class predictions
            class_scores = torch.softmax(logits, dim=1)
            
            # Get prototype weights for each class
            prototype_class_weights = self.model.last_layer.weight  # [num_classes, num_prototypes]
            
            return {
                'logits': logits,
                'class_scores': class_scores,
                'prototype_activations': prototype_activations,
                'prototype_distances': distances,
                'prototype_class_weights': prototype_class_weights,
            }