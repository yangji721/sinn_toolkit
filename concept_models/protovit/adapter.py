import torch
import torch.nn as nn
from ..base import ConceptModel
from .model import construct_PPNet

class ProtoVitModelAdapter(ConceptModel):
    """Adapter for ProtoViT model implementing the unified concept model interface"""
    
    def __init__(self, base_model=None, base_architecture='deit_small_patch16_224', 
                 num_classes=200, prototype_shape=(1000, 384, 1, 1), 
                 pretrained=True, img_size=224):
        super().__init__()
        
        if base_model is None:
            self.model = construct_PPNet(
                base_architecture=base_architecture,
                pretrained=pretrained,
                img_size=img_size, 
                prototype_shape=prototype_shape,
                num_classes=num_classes,
                sig_temp=1.0,
                radius=3
            )
        else:
            self.model = base_model
            
        self.num_classes = num_classes
        self.prototype_shape = prototype_shape
        
    def forward(self, x):
        """Forward pass returning class logits"""
        logits, min_distances, values = self.model(x)
        return logits
    
    def get_concepts(self, x):
        """Extract prototype-based concepts from input"""
        # Get prototype distances and feature maps
        conv_features, distances = self.model.subpatch_dist(x)
        
        # Get prototype activations through sigmoid
        slots = torch.sigmoid(self.model.patch_select * self.model.temp)
        
        # Compute prototype activation maps
        max_activations, min_distances, values = self.model.greedy_distance(x)
        
        return {
            'prototype_activations': max_activations,
            'distances': min_distances,
            'values': values,
            'feature_maps': conv_features
        }
        
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
            optimizer = torch.optim.Adam(self.model.parameters())
        
        best_acc = 0
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                    
                optimizer.zero_grad()
                output = self(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            train_acc = 100. * correct / total
            train_loss = train_loss / (batch_idx + 1)
            
            # Validation if loader provided
            if val_loader is not None:
                val_acc = self.evaluate(val_loader)
                if val_acc > best_acc:
                    best_acc = val_acc
                    
                print(f'Epoch {epoch}: Train Loss: {train_loss:.3f} | '
                      f'Train Acc: {train_acc:.3f}% | Val Acc: {val_acc:.3f}%')
            else:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}%')
            
            if scheduler is not None:
                scheduler.step()
                
    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                    
                output = self(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        accuracy = 100. * correct / total
        return accuracy
    
    def interpret(self, x):
        """Get interpretable concepts and their contributions"""
        # Get prototype activation maps and distances
        concepts = self.get_concepts(x)
        
        # Get final class predictions
        logits = self(x)
        class_scores = torch.softmax(logits, dim=1)
        
        # Get explanation maps
        explanations = {}
        explanations['concept_scores'] = concepts['prototype_activations']
        explanations['concept_maps'] = concepts['feature_maps']
        explanations['class_scores'] = class_scores
        explanations['distances'] = concepts['distances']
        explanations['patch_contributions'] = concepts['values']
        
        return explanations