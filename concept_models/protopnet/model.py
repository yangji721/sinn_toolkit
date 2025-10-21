import torch
import torch.nn as nn
import torch.nn.functional as F

class PPNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 num_classes, init_weights=True):
        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        # Initialize prototype vectors
        self.prototype_vectors = nn.Parameter(torch.rand(prototype_shape),
                                            requires_grad=True)
        
        # Initialize feature extractor (features is a function that returns the model)
        self.features = features(pretrained=True)
        
        # Get the expected feature dimensions
        if hasattr(self.features, 'expected_features_shape'):
            feature_channels = self.features.expected_features_shape[1]
        else:
            # Default for ResNet50: 2048
            feature_channels = 2048
        
        # Add 1x1 conv layer after feature extraction
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(feature_channels, prototype_shape[1], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(prototype_shape[1], prototype_shape[1], kernel_size=1),
            nn.ReLU()
        )
        
        # Last layer - prototype classification
        self.last_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Feature extraction
        features = self.features(x)
        features = self.add_on_layers(features)
        
        # Compute distances to prototypes
        prototype_activations = self._compute_distances(features)
        
        # Classification
        logits = self.last_layer(prototype_activations.view(batch_size, -1))
        
        return logits, prototype_activations
    
    def _compute_distances(self, x):
        # Compute squared L2 distances between features and prototypes in a conv-efficient way:
        # For each prototype p and spatial location, ||x - p||^2 = ||x||^2 + ||p||^2 - 2 * (x . p)
        # x: (B, C, H, W)
        # prototype_vectors: (P, C, 1, 1)
        batch_size = x.shape[0]
        P, Cp, _, _ = self.prototype_vectors.shape

        # ensure channel dims match
        if Cp != x.size(1):
            raise RuntimeError(f"Prototype channel dim ({Cp}) does not match feature channels ({x.size(1)})")

        # reshape prototypes for conv weight: (P, C, 1, 1)
        proto = self.prototype_vectors.view(P, Cp, 1, 1)

        # x_sq: (B, 1, H, W)
        x_sq = (x ** 2).sum(dim=1, keepdim=True)

        # p_sq: (1, P, 1, 1)
        p_sq = (proto.view(P, Cp) ** 2).sum(dim=1).view(1, P, 1, 1)

        # x_dot_p: (B, P, H, W)
        x_dot_p = F.conv2d(x, proto)

        # squared distances: (B, P, H, W)
        d2 = x_sq + p_sq - 2.0 * x_dot_p

        # numerical stability: clamp small negative values
        d2 = torch.clamp(d2, min=0.0)

        # global min pooling over spatial dims to get minimal distance per prototype
        # min_dists: (B, P)
        min_dists = -F.max_pool2d(-d2, kernel_size=(d2.size(2), d2.size(3))).view(batch_size, P)

        # convert to similarity score (higher = more similar)
        similarities = -min_dists
        return similarities
    
    def push_forward(self, x):
        """Get prototype activations."""
        features = self.features(x)
        features = self.add_on_layers(features)
        return self._compute_distances(features)
        
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        # Initialize the last layer with small weights and zero bias
        self.last_layer.weight.data.normal_(0, 0.01)