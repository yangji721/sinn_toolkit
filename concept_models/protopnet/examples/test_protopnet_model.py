import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from concept_models.protopnet.adapter import ProtoPNetModelAdapter
from concept_models.protopnet.settings import base_architecture, img_size, prototype_shape, num_classes

def test_protopnet_model():
    """Test ProtoPNet model with unified interface"""
    
    # Create model
    model = ProtoPNetModelAdapter(
        base_architecture=base_architecture,
        num_classes=num_classes,
        prototype_shape=prototype_shape,
        img_size=img_size
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train(False)  # Set to eval mode
    
    # Create sample input
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create a random test image
    img_tensor = torch.randn(1, 3, img_size, img_size)
    img_tensor = img_tensor.to(device)
    
    # Test forward pass
    with torch.no_grad():
        try:
            print("Testing ProtoPNet model:")
            print("Device:", device)
            print("Input shape:", img_tensor.shape)
            
            # Get model predictions
            logits = model(img_tensor)
            print(f"Model output shape: {logits.shape}")
            
            # Get predicted class
            pred_class = torch.argmax(logits).item()
            print(f"Predicted class: {pred_class}")
            
            # Get interpretable concepts
            interpretations = model.interpret(img_tensor)
            
            # Print interpretation results
            print("\nInterpretation Results:")
            print("Class scores shape:", interpretations['class_scores'].shape)
            print("Prototype activations shape:", interpretations['prototype_activations'].shape)
            print("Prototype distances shape:", interpretations['prototype_distances'].shape)
            print("Prototype-class weights shape:", interpretations['prototype_class_weights'].shape)
            
        except Exception as e:
            print("\nError during model execution:")
            print(f"Exception: {str(e)}")
            raise

if __name__ == '__main__':
    test_protopnet_model()