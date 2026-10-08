"""Dilated TCN: strided stem + residual dilated convolutions, one logit every `stem_stride` samples.

Details: README.md in this folder.
"""

from .model import DilatedTCN

Model = DilatedTCN

# Small hyper-parameters for the fast automatic contract test (tests/test_model_contract.py).
SMOKE_PARAMS = {"hidden": 16, "levels": 3}

__all__ = ["DilatedTCN", "Model", "SMOKE_PARAMS"]
