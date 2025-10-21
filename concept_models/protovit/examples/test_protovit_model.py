import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
# Ensure upstream ProtoViT directory is in sys.path for imports
PROTOVIT_UPSTREAM = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ProtoViT'))
if PROTOVIT_UPSTREAM not in sys.path:
    sys.path.insert(0, PROTOVIT_UPSTREAM)

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
    
def train_demo_protovit(adapter: ProtoVitModelAdapter, device, img_size=224, epochs=2, batch_size=8, lr=1e-3):
    """Tiny synthetic training demo for ProtoViT.

    Probes the adapter's underlying model for input channels and classes, then trains for a few epochs
    on random data to demonstrate the training loop.
    """
    # Get underlying module (adapter may wrap the model in DataParallel)
    mm = adapter.model.module if hasattr(adapter.model, 'module') else adapter.model
    mm.train()

    import torch.nn as nn
    in_ch = 3
    for layer in mm.modules():
        if isinstance(layer, nn.Conv2d):
            in_ch = layer.in_channels
            break

    # infer number of classes by a single forward
    with torch.no_grad():
        sample = torch.randn(1, in_ch, img_size, img_size).to(device)
        out = mm(sample)
        if isinstance(out, (tuple, list)):
            out = out[0]
        n_classes = out.shape[1]

    from torch.utils.data import TensorDataset, DataLoader
    N = 64
    X = torch.randn(N, in_ch, img_size, img_size)
    y = torch.randint(0, n_classes, (N,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optim = torch.optim.Adam(mm.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            out = mm(xb)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.size(0)
            _, preds = logits.max(1)
            total_correct += (preds == yb).sum().item()
            total += xb.size(0)
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/total:.4f}, acc: {total_correct/total:.4f}")


if __name__ == '__main__':
    # Run the small ProtoViT tests and then a tiny synthetic training demo
    test_protovit_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = ProtoVitModelAdapter(base_architecture='deit_small_patch16_224', num_classes=1000, prototype_shape=(1000,384,1,1), pretrained=False)
    adapter = adapter.to(device) if hasattr(adapter, 'to') else adapter
    try:
        print('\nRunning tiny synthetic training demo for ProtoViT...')
        train_demo_protovit(adapter, device, img_size=224, epochs=2, batch_size=8)
    except Exception as e:
        print('ProtoViT training demo failed:', e)