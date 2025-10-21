"""
This module contains the main public API of the bcos package.
"""
"""Adapter wrapper for the external B-cos package.

This module intentionally avoids forcing imports of a top-level ``bcos``
package during package import. Instead, it attempts to resolve the installed
``bcos`` package at runtime. This allows developers to keep the original
development copy of ``B-cos-v2`` in the repository while still using the
installed/packaged import mechanism (``import bcos`` / ``torch.hub``).

If you want to use the adapter that wraps the installed package, import
``concept_models.bcos`` and the module will try to import the system-level
``bcos`` package lazily. If that fails, a clear ImportError will be raised.
"""

try:
    # Prefer the installed/available top-level package. This allows using
    # the developer-installed package (e.g. pip install -e B-cos-v2) or
    # a regular installation.
    import bcos as _bcos
except Exception as _e:
    # Re-raise with a clearer message while preserving the original traceback
    raise ImportError(
        "Unable to import top-level `bcos` package.\n"
        "If you want to use the bundled B-cos development copy, either install it "
        "in editable mode (pip install -e /path/to/B-cos-v2) or ensure its parent "
        "directory is on PYTHONPATH. Original error: {}".format(_e)
    )

# Expose commonly used submodules from the installed package via simple names
presets = _bcos.data.presets
transforms = _bcos.data.transforms
models = _bcos.models
pretrained = _bcos.models.pretrained
modules = _bcos.modules
optim = _bcos.optim
settings = _bcos.settings

# Extract commonly used utilities from the imported top-level `bcos` module
BcosUtilMixin = _bcos.common.BcosUtilMixin
explanation_mode = _bcos.common.explanation_mode
gradient_to_image = _bcos.common.gradient_to_image
plot_contribution_map = _bcos.common.plot_contribution_map

# version
__version__ = _bcos.version.__version__

__all__ = [
    "presets",
    "transforms",
    "models",
    "pretrained",
    "modules",
    "optim",
    "settings",
    "BcosUtilMixin",
    "explanation_mode",
    "gradient_to_image",
    "plot_contribution_map",
]

# Convenience API for external package
from .api import create_model, list_pretrained, train  # noqa: F401
__all__.extend(["create_model", "list_pretrained", "train"])
