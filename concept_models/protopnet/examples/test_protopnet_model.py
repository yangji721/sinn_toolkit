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

    # ----------------- Small training demo -----------------
    def train_demo(model, device, epochs=2, batch_size=8, lr=1e-3):
        """Tiny training loop on synthetic data to demonstrate API."""
        model.train()

        # infer input channels
        mm = model.module if hasattr(model, 'module') else model
        in_ch = 3
        import torch.nn as nn
        for layer in mm.modules():
            if isinstance(layer, nn.Conv2d):
                in_ch = layer.in_channels
                break

        # infer number of classes via forward
        with torch.no_grad():
            sample = torch.randn(1, in_ch, img_size, img_size).to(device)
            out = model(sample)
            if isinstance(out, (tuple, list)):
                out = out[0]
            n_classes = out.shape[1]

        # synthetic dataset
        N = 64
        X = torch.randn(N, in_ch, img_size, img_size)
        y = torch.randint(0, n_classes, (N,))
        from torch.utils.data import TensorDataset, DataLoader
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                outputs = model(xb)
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
                _, preds = logits.max(1)
                total_correct += (preds == yb).sum().item()
                total += xb.size(0)

            print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/total:.4f}, acc: {total_correct/total:.4f}")

    # run a short demo if invoked as script
    if __name__ == '__main__':
        # create model again for training demo to ensure fresh state
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_for_train = ProtoPNetModelAdapter(
            base_architecture=base_architecture,
            num_classes=num_classes,
            prototype_shape=prototype_shape,
            img_size=img_size
        ).to(device)
        try:
            train_demo(model_for_train, device, epochs=2, batch_size=8, lr=1e-3)
        except Exception as e:
            print('Training demo failed:', e)