# Self-Interpretable Neural Network Toolkit

This repository offers a compact toolkit that integrates various self-interpretable model implementations. It provides a consistent adapter interface for easy experimentation and integration. The goal is to simplify the use of prototype-, concept-, and attribution-based models through a unified API while maintaining the original models' provenance.

## Key Features

* **Unified Adapter Interfaces**: The toolkit includes adapters for the following models under `concept_models/*`:

  * **`bcos` (B-cos)**: Adapter for the B-cos package (recommended via editable install)
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

### Notes about `bcos` and This Repository

* `concept_models.bcos` serves as an adapter that attempts to import the top-level `bcos` package (either the installed or editable package). This keeps the adapter lightweight while allowing you to develop within the original `B-cos-v2` repository.

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

* **B-cos (B-cos toolkit)**
  Original Repository: `B-cos-v2` (this repo contains a copy)
  Paper: ["B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers"](https://ieeexplore.ieee.org/document/9736579), TPAMI 2024.

* **ProtoPNet**
  Original Code: ProtoPNet-Concept_final
  Paper: ["This Looks Like That: Deep Learning for Interpretable Image Recognition"](https://papers.nips.cc/paper/8843-this-looks-like-that-deep-learning-for-interpretable-image-recognition), NeurIPS 2019.

* **ProtoPool**
  Original Code: ProtoPool-Concept_final
  Paper: ["ProtoPool: Interpretable Image Classification with Differentiable Prototypes Assignment"](https://openaccess.thecvf.com/content/ECCV_2022/html/Yu_ProtoPool_Interpretable_Image_Classification_With_Differentiable_Prototypes_Assignment_ECCV_2022_paper.html), ECCV 2022.

* **TesNet**
  Original Code: TesNet-Concept_final
  Paper: ["Interpretable Image Recognition by Constructing Transparent Embedding Space"](https://openaccess.thecvf.com/content/ICCV_2021/html/Liu_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.html), ICCV 2021.

* **ProtoViT**
  Original Code: ProtoViT
  Paper: ["Interpretable Image Classification with Adaptive Prototype-based Vision Transformers"](https://papers.nips.cc/paper/10704-interpretable-image-classification-with-adaptive-prototype-based-vision-transformers), NeurIPS 2024.

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

---

This version cleans up the content for better readability and organization, while emphasizing the core details. Let me know if you need further adjustments!
