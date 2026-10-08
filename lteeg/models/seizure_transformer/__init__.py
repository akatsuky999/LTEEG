"""SeizureTransformer (Wu et al., 2025): U-shaped CNN with a Transformer bottleneck, one logit per sample.

Details: README.md in this folder.
"""

from .model import SeizureTransformer, load_original_state_dict

Model = SeizureTransformer

# Small hyper-parameters for the fast automatic contract test (tests/test_model_contract.py).
SMOKE_PARAMS = {"num_layers": 1, "dim_feedforward": 64}

__all__ = ["Model", "SeizureTransformer", "SMOKE_PARAMS", "load_original_state_dict"]
