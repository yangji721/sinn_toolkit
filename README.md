# Self-interpretable Models Toolkit

This repository provides a compact toolkit that unifies several self-interpretable
model implementations and exposes a small, consistent adapter interface for
experimentation and integration. The goal is to make different prototype-, concept- and
attribution-based models easy to import and use via a common API while preserving
the original implementations' provenance.

Key features in this repository
- Unified adapter interfaces under `concept_models/*` for the following models:
	- `bcos` (B-cos): adapter to the B-cos package (preferred via editable install)
	- `protopnet` (ProtoPNet)
	- `protopool` (ProtoPool)
	- `tesnet` (TesNet)
	- `protovit` (ProtoViT)

Each adapter exposes a small contract (example in `concept_models/base.py`):
- `forward(x)` — standard forward to produce logits
- `train(...)` — training helper that wraps the model's training routines
- `interpret(x)` / `get_concepts(x)` — return prototype/concept activations and
	other interpretability artifacts

Installation

1. Recommended: install the original B-cos development package in editable
	 mode so the top-level `bcos` import resolves to the development copy:

```bash
pip install -e .
```

2. Install Python runtime dependencies (top-level):

```bash
pip install -r requirements.txt
```

3. If you prefer to use the PyPI release of `bcos`, install it via pip:

```bash
pip install bcos
```

Notes about `bcos` and this repository
- `concept_models.bcos` acts as an adapter that tries to import the top-level
	`bcos` package (the installed or editable package). This keeps the adapter
	light while letting you develop inside the original `B-cos-v2` repository.

Quick usage example

```python
from concept_models.bcos import presets
from concept_models.bcos.api import create_model, list_pretrained

# Example: list pretrained entrypoints (requires bcos installed/editable)
print(list_pretrained()[:10])

# Create a model via the adapter factory
model = create_model('resnet50', pretrained_weights=True)
```

Provenance and papers
- B-cos (B-cos toolkit) — original repository: `B-cos-v2` (this repo contains a
	copy). Paper: "B-cos Alignment for Inherently Interpretable CNNs and Vision
	Transformers", TPAMI 2024.
- ProtoPNet — original code: ProtoPNet-Concept_final. Paper: "This Looks Like
	That: Deep Learning for Interpretable Image Recognition", NeurIPS 2019.
- ProtoPool — original code: ProtoPool-Concept_final. Paper: "ProtoPool:
	Interpretable Image Classification with Differentiable Prototypes Assignment",
	ECCV 2022.
- TesNet — original code: TesNet-Concept_final. Paper: "Interpretable Image
	Recognition by Constructing Transparent Embedding Space", ICCV 2021.
- ProtoViT — original code: ProtoViT. Paper: "Interpretable Image
	Classification with Adaptive Prototype-based Vision Transformers", NeurIPS 2024.

Files changed in this repository to produce the unified adapters
- `concept_models/*` — adapter packages for each model with `adapter.py`,
	`models/` (local features as needed), `util/` and `examples/`.

Running examples

```bash
# from repository root
python concept_models/bcos/examples/test_bcos_model.py
python concept_models/tesnet/examples/test_tesnet_model.py
python concept_models/protopnet/examples/test_protopnet_model.py
```

Requirements (top-level)
- See `requirements.txt` (this repo). Key runtime requirements:
	- torch>=1.13
	- torchvision
	- numpy
	- pillow
	- einops
	- bcos (recommended: pip install -e B-cos-v2 or pip install bcos)

If you maintain a conda environment, use the provided environment yml files in
the upstream projects where available.

License

This project combines code from several upstream repositories. Each module
retains its original license; the repository's top-level code is provided under
the Apache-2.0 license. When reusing code from papers, please follow their
respective license and citation guidelines.