"""Template model package (not a model: folders starting with '_' are skipped).

Every model package exports ``Model`` and, optionally, ``SMOKE_PARAMS``.
"""

from .model import TemplateNet

Model = TemplateNet

# Small hyper-parameters used by the automatic contract test (tests/test_model_contract.py).
SMOKE_PARAMS = {"hidden": 8, "depth": 1}

__all__ = ["Model", "SMOKE_PARAMS", "TemplateNet"]
