"""Example script demonstrating how to use the B-cos model with the unified interface.

This script shows how to:
1. Load a pretrained B-cos model
2. Process input images
3. Get predictions
4. Extract concept information
5. Generate and visualize interpretations
"""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from concept_models.base import ConceptModel
from concept_models.bcos.api import create_model, list_pretrained, train
from bcos.data.presets import ImageNetClassificationPresetEval
from bcos.common import explanation_mode, gradient_to_image, plot_contribution_map
from PIL import Image
import numpy as np

# For this example we'll use the factory to obtain a model adapter directly.
# Use ImageNetClassificationPresetEval with a crop_size matching the model (224 for resnet50).
PRESET = ImageNetClassificationPresetEval(crop_size=224)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model using unified factory (supports pretrained entrypoints and experiments)
    print("Available pretrained entrypoints (subset):", list_pretrained()[:10])
    print("Creating B-cos model via factory...")
    model = create_model('resnet50', pretrained_weights=True)
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    # Prefer using the model's attached transform (pretrained entrypoints attach config.transform)
    from PIL import Image as PILImage
    pil = PILImage.new("RGB", (256, 256), color=(128, 128, 128))
    if hasattr(model, 'transform') and model.transform is not None:
        img = model.transform(pil).unsqueeze(0)
    else:
        # fallback to the preset defined above (ensures AddInverse is applied if is_bcos=True)
        img = PRESET(pil).unsqueeze(0)
    img = img.to(device)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(img)
        pred_class = output.argmax(dim=1).item()
        print(f"Output shape: {output.shape}")
        print(f"Predicted class: {pred_class}")
        
    print("\nTesting concept extraction...")
    # If model implements our wrapper API, call get_concepts; otherwise try B-cos model utilities
    if hasattr(model, 'get_concepts'):
        with torch.no_grad():
            concepts = model.get_concepts(img)
    else:
        with explanation_mode():
            model(img)
            concepts = {'concept_scores': model.get_last_alignment() if hasattr(model, 'get_last_alignment') else None}

    print("Concept information:")
    for key, value in concepts.items():
        if value is not None:
            try:
                print(f"- {key} shape: {value.shape}")
            except Exception:
                print(f"- {key}: {type(value)}")
            
    print("\nTesting interpretation...")
    # If the model exposes interpret(), use it; otherwise fall back to manual explanation
    if hasattr(model, 'interpret'):
        interpretation = model.interpret(img)
    else:
        with explanation_mode():
            img.requires_grad_(True)
            out = model(img)
            pred = out.argmax(dim=1)
            out[:, pred].sum().backward()
            contrib_map = gradient_to_image(img.grad)
            alignment = model.get_last_alignment() if hasattr(model, 'get_last_alignment') else None
            interpretation = {
                'prediction': pred,
                'contribution_map': contrib_map,
                'concept_scores': alignment,
                'explanation': None,
            }
    print("Interpretation results:")
    for key, value in interpretation.items():
        if isinstance(value, torch.Tensor):
            print(f"- {key} shape: {value.shape}")
    
    # Visualize interpretation if available
    if isinstance(interpretation.get('explanation'), plt.Figure):
        plt.show()
    else:
        print("Skipping visualization for random input")

    # Demonstrate how to start a dry-run training using existing bcos trainer (no heavy work)
    print("\nDemonstrating train() dry-run with minimal args (will prepare config but not run full training):")
    try:
        train_args = dict(
            base_directory='./experiments_demo',
            dataset='ImageNet',
            base_network='bcos_final',
            experiment_name='resnet_50',
            fast_dev_run=True,
            resume=False,
        )
        # call train() - this will trigger trainer.run_training; fast_dev_run=True keeps it lightweight
        # WARNING: this may attempt to import heavy modules or access datasets; run explicitly if desired.
        print("Train helper prepared. Call train(train_args) to execute a dev training run.")
    except Exception as e:
        print("Train helper not executed in example (to avoid heavy operations)", e)

if __name__ == '__main__':
    main()