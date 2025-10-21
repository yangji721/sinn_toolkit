# Self-Interpretable Neural Network Toolkit

This repository offers a compact toolkit that integrates various self-interpretable model implementations. It provides a consistent adapter interface for easy experimentation and integration. The goal is to simplify the use of prototype-, concept-, and attribution-based models through a unified API while maintaining the original models' provenance.

## Key Features

* **Unified Adapter Interfaces**: The toolkit includes adapters for the following models under `concept_models/*`:

  * **`bcos` (B-cos)**
  * **`protopnet` (ProtoPNet)**
  * **`protopool` (ProtoPool)**
  * **`tesnet` (TesNet)**
  * **`protovit` (ProtoViT)**

Each adapter exposes a small contract as defined in `concept_models/base.py`:

* `forward(x)` — Standard forward function to produce logits
* `train(...)` — A helper for training, wrapping the model's training routines
* `interpret(x)` / `get_concepts(x)` — Returns prototype/concept activations and other interpretability artifacts

## Installation

1. **Recommended**: Install the original B-cos development package in editable mode so the top-level `bcos` import resolves to the development copy:

   ```bash
   pip install -e .
   ```

2. Install Python runtime dependencies (top-level):

   ```bash
   pip install -r requirements.txt
   ```

## Quick Usage Example

```python
from concept_models.bcos import presets
from concept_models.bcos.api import create_model, list_pretrained

# Example: List pretrained entrypoints (requires bcos installed/editable)
print(list_pretrained()[:10])

# Create a model via the adapter factory
model = create_model('resnet50', pretrained_weights=True)
```

## Provenance and Papers

* **B-cos**
  Paper: ["B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers"](https://ieeexplore.ieee.org/iel7/34/10522060/10401936.pdf), TPAMI 2024.

* **ProtoPNet**
  Paper: ["This Looks Like That: Deep Learning for Interpretable Image Recognition"](http://papers.neurips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf), NeurIPS 2019.

* **ProtoPool**
  Paper: ["ProtoPool: Interpretable Image Classification with Differentiable Prototypes Assignment"](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720346.pdf), ECCV 2022.

* **TesNet**
  Paper: ["Interpretable Image Recognition by Constructing Transparent Embedding Space"](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf), ICCV 2021.

* **ProtoViT**
  Paper: ["Interpretable Image Classification with Adaptive Prototype-based Vision Transformers"](https://neurips.cc/virtual/2024/poster/94047), NeurIPS 2024.

## Files Changed in This Repository to Produce the Unified Adapters

* **`concept_models/*`**: Adapter packages for each model, including `adapter.py`, `models/` (local features), `util/`, and `examples/`.

## Running Examples

```bash
# From the repository root
python concept_models/bcos/examples/test_bcos_model.py
python concept_models/tesnet/examples/test_tesnet_model.py
python concept_models/protopnet/examples/test_protopnet_model.py
```

## Requirements (Top-Level)

See `requirements.txt` for detailed dependencies. Key runtime requirements include:

* `torch>=1.13`
* `torchvision`
* `numpy`
* `pillow`
* `einops`
* `bcos` (recommended: `pip install -e B-cos-v2` or `pip install bcos`)

If you maintain a conda environment, use the provided environment `.yml` files from upstream projects where available.

## License

This project integrates code from several upstream repositories. Each module retains its original license. The repository's top-level code is provided under the Apache-2.0 license. When reusing code from papers, please adhere to their respective license and citation guidelines.