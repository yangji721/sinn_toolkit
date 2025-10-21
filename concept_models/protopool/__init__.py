"""ProtoPool package helpers and adapter exports.

Avoid importing the heavy `model_cap_final` module at package import time to
prevent module resolution issues for local feature modules. Import the
implementation lazily inside the adapter when needed.
"""

from .adapter import ProtoPoolAdapter

__all__ = ["ProtoPoolAdapter"]