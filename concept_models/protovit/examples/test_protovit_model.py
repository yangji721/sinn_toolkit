import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from concept_models.protovit.adapter import ProtoVitModelAdapter

def test_protovit_model():
    """Test ProtoViT model with unified interface"""
    
    # Create model with default settings
    model = ProtoVitModelAdapter(
        base_architecture='deit_small_patch16_224',
        num_classes=1000,  # ImageNet classes
        prototype_shape=(1000, 384, 1, 1),  # Match DeiT-small feature dimensions
        pretrained=True
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Create sample input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create a random test image
    img_tensor = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
    
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # Set model to evaluation mode using the standard PyTorch way
    model.train(False)  # This calls our modified train method with mode=False
    img_tensor = img_tensor.to(device)
    
    # Test forward pass
    with torch.no_grad():
        try:
            # Print model state
            print("Device:", device)
            print("Input shape:", img_tensor.shape)
            print("Input device:", img_tensor.device)
            
            logits = model(img_tensor)
            print(f"Model output shape: {logits.shape}")
            
            # Get predicted class
            pred_class = torch.argmax(logits).item()
            print(f"Predicted class: {pred_class}")
            
            # Get interpretable concepts
            interpretations = model.interpret(img_tensor)
            
            # Print interpretation results
            print("\nInterpretation Results:")
            print(f"Number of prototypes: {interpretations['concept_scores'].shape[1]}")
            print(f"Feature map shape: {interpretations['concept_maps'].shape}")
            print(f"Number of classes: {interpretations['class_scores'].shape[1]}")
            print(f"Patch contribution shape: {interpretations['patch_contributions'].shape}")
        except Exception as e:
            print("Error during model execution:")
            print("Exception:", str(e))
            raise
        
if __name__ == '__main__':
    test_protovit_model()