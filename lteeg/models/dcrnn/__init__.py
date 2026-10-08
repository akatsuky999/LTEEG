"""DCRNN (Li et al., 2018; Tang et al., 2022): diffusion-convolutional GRU over a channel graph, one logit per time step.

Details: README.md in this folder.
"""

from .model import DCRNN

Model = DCRNN

# Small hyper-parameters for the fast automatic contract test (tests/test_model_contract.py).
SMOKE_PARAMS = {"num_rnn_layers": 1, "rnn_units": 8, "top_k": 2}

__all__ = ["DCRNN", "Model", "SMOKE_PARAMS"]
