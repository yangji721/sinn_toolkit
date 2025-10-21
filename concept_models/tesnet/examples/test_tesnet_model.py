import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

sys.path.append('/data/yang/benchmark/SINN')
from concept_models.tesnet.adapter import TesNetAdapter

def train_demo_tesnet(adapter: TesNetAdapter, device, img_size=224, epochs=2, batch_size=8, lr=1e-3):
    """A tiny training demo that probes the adapter for input channel count and number of classes.

    If the adapter's underlying model expects non-RGB inputs, this function will generate synthetic
    inputs with the correct channel count before training.
    """
    # Use the adapter's internal model (DataParallel.module when wrapped) for module-level ops
    mm = adapter.model.module if hasattr(adapter.model, 'module') else adapter.model
    mm.train()

    import torch.nn as nn
    in_ch = 3
    for layer in mm.modules():
        if isinstance(layer, nn.Conv2d):
            in_ch = layer.in_channels
            break

    # infer classes by a single forward
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

def main():
    # Configuration
    img_size = 224
    num_classes = 200
    batch_size = 64
    num_prototypes = 2000
    feature_dimension = 128
    
    # Setup data transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load datasets (expected layout: data_dir/train and data_dir/test)
    data_dir = '/data/yang/benchmark/CUB_200_2011'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')

    # Create dataset loaders (this function assumes the dataset is present). The
    # top-level runner will decide whether to call main() or run the synthetic demo.
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Create TesNet model
    model = TesNetAdapter(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        feature_dimension=feature_dimension,
        img_size=img_size,
        base_architecture='resnet50',
        cap_width=6.0
    )
    
    # Training configurations
    coefs = {
        'crs_ent': 1.0,   # Cross entropy loss
        'clst': 0.8,      # Clustering loss
        'sep': 0.08,      # Separation loss  
        'l1': 1e-4,       # L1 regularization
        'orth': 1.0,      # Orthogonality loss
        'sub_sep': 1e-7,  # Subspace separation
        'cap_coef': 3e-3  # Capsule loss coefficient
    }

    # Train model
    print("Starting training...")
    metrics = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        warm_epochs=5,
        coefs=coefs
    )
    
    print(f"Training completed. Best validation accuracy: {metrics['best_val_acc']:.2f}%")
    
    # Save model
    save_path = 'tesnet_model.pth'
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # Example of interpretation
    # Get a batch of images from validation set
    images, labels = next(iter(val_loader))
    
    # Get interpretable components
    interpretations = model.interpret(images)
    
    # Print interpretation results
    print("\nInterpretation Results:")
    print(f"Logits shape: {interpretations['logits'].shape}")
    print(f"Prototype activations shape: {interpretations['prototype_activations'].shape}")
    print(f"Prototype similarities shape: {interpretations['prototype_similarities'].shape}")
    print(f"Concept cap widths shape: {interpretations['concept_cap_widths'].shape}")

if __name__ == '__main__':
    # Decide whether to run full example (requires CUB dataset) or a tiny synthetic demo
    data_dir = '/data/yang/benchmark/CUB_200_2011'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')
    has_data = os.path.isdir(train_dir) and os.path.isdir(val_dir)
    if has_data:
        # Run full example which will load real data and train (may take long)
        main()
    else:
        # Run a small synthetic demo instead
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        adapter = TesNetAdapter(num_classes=200, num_prototypes=2000, feature_dimension=128, img_size=224, device=dev)
        try:
            print('Running small synthetic training demo for TesNet...')
            train_demo_tesnet(adapter, dev, img_size=224, epochs=2, batch_size=8)
        except Exception as e:
            print('TesNet training demo failed:', e)