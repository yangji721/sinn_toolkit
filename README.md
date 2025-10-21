# 🌟 Self-Interpretable Neural Network Toolkit

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%9D%A4-red?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Interpretable AI](https://img.shields.io/badge/focus-interpretable%20models-orange.svg)]()

> 🧠 A compact toolkit for **self-interpretable neural networks** with a unified adapter interface.

This repository integrates multiple prototype-, concept-, and attribution-based interpretable model implementations into one consistent framework — simplifying experimentation while maintaining model provenance.

---

## 🗂️ Table of Contents
- [Key Features](#-key-features)
- [Installation](#️-installation)
- [Quick Usage Example](#-quick-usage-example)
- [Provenance and Papers](#-provenance-and-papers)
- [Running Examples](#-running-examples)
- [Requirements](#-requirements)
- [License](#-license)

---

## 🚀 Key Features

### 🔧 Unified Adapter Interfaces
Adapters for the following models are provided under `concept_models/*`:

- **`bcos`** — B-cos  
- **`protopnet`** — ProtoPNet  
- **`protopool`** — ProtoPool  
- **`tesnet`** — TesNet  
- **`protovit`** — ProtoViT  

Each adapter implements a minimal contract defined in `concept_models/base.py`:

| Method | Description |
|--------|--------------|
| `forward(x)` | Standard forward function producing logits |
| `train(...)` | Wrapper for model-specific training routines |
| `interpret(x)` / `get_concepts(x)` | Returns prototype/concept activations and interpretability artifacts |

---

## ⚙️ Installation

1. **Recommended:** install in editable mode so top-level imports (like `bcos`) resolve correctly:

```python
  pip install -e .
```

2. Install dependencies:
```python
  pip install -r requirements.txt
```

## 💡 Quick Usage Example

```python
from concept_models.bcos import presets
from concept_models.bcos.api import create_model, list_pretrained

# List pretrained entrypoints (requires bcos installed/editable)
print(list_pretrained()[:10])

# Create a model via the adapter factory
model = create_model('resnet50', pretrained_weights=True)
```

---

## 📚 Provenance and Papers

| Model         | Reference                                                                                                                                                                                                                                              |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **B-cos**     | [*B-cos Alignment for Inherently Interpretable CNNs and Vision Transformers*](https://ieeexplore.ieee.org/iel7/34/10522060/10401936.pdf), TPAMI 2024                                                                                                   |
| **ProtoPNet** | [*This Looks Like That: Deep Learning for Interpretable Image Recognition*](http://papers.neurips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf), NeurIPS 2019                                              |
| **ProtoPool** | [*ProtoPool: Interpretable Image Classification with Differentiable Prototypes Assignment*](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720346.pdf), ECCV 2022                                                                         |
| **TesNet**    | [*Interpretable Image Recognition by Constructing Transparent Embedding Space*](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf), ICCV 2021 |
| **ProtoViT**  | [*Interpretable Image Classification with Adaptive Prototype-based Vision Transformers*](https://neurips.cc/virtual/2024/poster/94047), NeurIPS 2024                                                                                                   |

---

## 🧪 Running Examples

```bash
# From repository root
python concept_models/bcos/examples/test_bcos_model.py
python concept_models/tesnet/examples/test_tesnet_model.py
python concept_models/protopnet/examples/test_protopnet_model.py
python concept_models/protopool/examples/test_protopool_model.py
python concept_models/protovit/examples/test_protovit_model.py
```

---

## 📦 Requirements

See `requirements.txt` for full dependency details.
Key runtime dependencies include:

* `torch>=1.13`
* `torchvision`
* `numpy`
* `pillow`
* `einops`
* `bcos`

> 💡 For **conda users**, upstream `.yml` environments are recommended where available.

---

## 📄 License

This project integrates code from several upstream repositories.
Each submodule retains its **original license**.
Top-level code is distributed under **Apache-2.0**.

When reusing code from research papers, please follow their respective **license and citation** guidelines.


