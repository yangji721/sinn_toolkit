"""Concept Models Package - lightweight top-level import.

We intentionally avoid importing all subpackages at package import time because
some of the third-party model folders (e.g. ProtoPNet) contain top-level
imports that assume a different module layout (non-relative imports like
``import resnet_features``). Importing those subpackages on package import
causes ModuleNotFoundError. Import the specific subpackage explicitly when
needed.

Use:
	from concept_models import bcos
or
	import concept_models.bcos as bcos

Other subpackages (protopnet, protopool, ...) can be imported directly when
you actually need them.
"""

 # Do NOT import subpackages at top-level. Import them explicitly when needed to
 # avoid ModuleNotFoundError caused by third-party code that expects a
 # different module layout. Example usage:
 #
 #   import concept_models.bcos as bcos_adapter
 #   from concept_models.protopnet import ProtoPNetAdapter
 #
 # The `bcos` name below was removed to prevent shadowing an installed
 # top-level `bcos` package. If you want the B-cos adapter, import
 # `concept_models.bcos` explicitly.

__version__ = "0.1.0"