import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

sys.path.append('/data/yang/benchmark/SINN')
from concept_models.tesnet.adapter import TesNetAdapter

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

    # Load datasets
    data_dir = '/data/yang/benchmark/CUB_200_2011'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test')
    
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
    main()