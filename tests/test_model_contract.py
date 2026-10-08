"""Every model package must satisfy the contract in lteeg/models/base.py.

New model folders are picked up automatically: adding lteeg/models/<name>/ is enough
for it to be tested here (with its SMOKE_PARAMS).
"""

import importlib
import sys

import pytest
import torch

from lteeg.config import CHBMIT_CHANNELS, load_config
from lteeg.models import (
    PACKAGE_DIR,
    available_models,
    build_model,
    describe_available_models,
    get_model_class,
    instantiate_model,
    load_weights,
    smoke_params,
)
from lteeg.models.contract import ModelContractError, check_contract
from lteeg.registry import RegistryError

IN_SAMPLES = 2048  # 8 s at 256 Hz: short enough for CPU tests, long enough for every model
CONTEXT = {"in_channels": 18, "in_samples": IN_SAMPLES, "fs": 256.0, "channel_names": list(CHBMIT_CHANNELS)}


def test_builtin_models_are_discovered():
    names = available_models()
    assert {"seizure_transformer", "tcn", "dcrnn"} <= set(names)
    assert not any(n.startswith("_") for n in names)  # _template is not a model


@pytest.mark.parametrize("name", available_models())
def test_model_package_layout(name):
    folder = PACKAGE_DIR / name
    for rel in ("__init__.py", "model.py", "layers/__init__.py", "README.md"):
        assert (folder / rel).is_file(), f"lteeg/models/{name}/{rel} is missing"
    module = importlib.import_module(f"lteeg.models.{name}")
    assert isinstance(module.Model, type) and issubclass(module.Model, torch.nn.Module)
    assert isinstance(getattr(module, "SMOKE_PARAMS", None), dict), "declare small SMOKE_PARAMS for fast tests"
    assert (module.__doc__ or "").strip(), "the package docstring's first line is shown by `lteeg list-models`"


@pytest.mark.parametrize("num_outputs", [1, 3])
@pytest.mark.parametrize("name", available_models() + ["_template"])
def test_model_contract(name, num_outputs):
    module = importlib.import_module(f"lteeg.models.{name}")
    params = getattr(module, "SMOKE_PARAMS", {})
    context = {**CONTEXT, "num_outputs": num_outputs}
    report = check_contract(lambda: instantiate_model(module.Model, name, context, params), 18, IN_SAMPLES,
                            num_outputs, batch_size=3)
    assert report.output_shape[:2] == (3, num_outputs)


@pytest.mark.parametrize("name", available_models())
def test_default_parameters_build_from_config(name):
    cfg = load_config(None, [f"model.name={name}", "model.params={}"])
    model = build_model(cfg, 1)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_describe_available_models_lists_tunable_params():
    rows = {r["name"]: r for r in describe_available_models()}
    assert "fs" in rows["dcrnn"]["context"] and "rnn_units" in rows["dcrnn"]["params"]
    assert "in_channels" not in rows["seizure_transformer"]["params"]


def test_smoke_params_accessor():
    assert smoke_params("tcn") == {"hidden": 16, "levels": 3}


# ----------------------------------------------------------------------------- lookup & errors
def test_unknown_model_suggests_closest_name():
    with pytest.raises(RegistryError, match="did you mean 'dcrnn'"):
        get_model_class("dcrn")


def test_misspelled_parameter_is_rejected_with_hint():
    cfg = load_config(None, ["model.name=dcrnn", "model.params={rnn_unit: 8}"])
    with pytest.raises(RegistryError, match="'rnn_unit' -> 'rnn_units'"):
        build_model(cfg, 1)


def test_framework_provided_arguments_cannot_be_set_in_params():
    cfg = load_config(None, ["model.name=tcn", "model.params={in_channels: 4}"])
    with pytest.raises(RegistryError, match="must not set"):
        build_model(cfg, 1)


EXTERNAL = '''
import torch.nn as nn

class WithContext(nn.Module):
    def __init__(self, in_channels, in_samples, num_outputs, fs, channel_names, width=4):
        super().__init__()
        self.fs, self.channel_names = fs, channel_names
        self.conv = nn.Conv1d(in_channels, num_outputs, 1)

    def forward(self, x):
        return self.conv(x)

class Plain(nn.Module):
    def __init__(self, in_channels, in_samples, num_outputs):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, num_outputs, 1)

    def forward(self, x):
        return self.conv(x)

class MissingArgs(nn.Module):
    def __init__(self, in_channels, num_outputs):
        super().__init__()
'''


@pytest.fixture
def external_module(tmp_path, monkeypatch):
    pkg = tmp_path / "my_ext"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "nets.py").write_text(EXTERNAL)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "my_ext.nets"
    for mod in ("my_ext.nets", "my_ext"):
        sys.modules.pop(mod, None)


def test_external_model_and_optional_context(external_module):
    cfg = load_config(None, [f"model.name={external_module}:WithContext", "model.params={width: 8}"])
    m = build_model(cfg, 1)
    assert m.fs == cfg.fs and m.channel_names == list(cfg.data.channels)
    plain = build_model(load_config(None, [f"model.name={external_module}:Plain", "model.params={}"]), 1)
    assert isinstance(plain, torch.nn.Module)  # fs/channel_names not passed: not declared


def test_external_model_errors(external_module):
    with pytest.raises(RegistryError, match="must accept"):
        build_model(load_config(None, [f"model.name={external_module}:MissingArgs", "model.params={}"]), 1)
    with pytest.raises(RegistryError, match="has no attribute"):
        get_model_class(f"{external_module}:Nope")
    with pytest.raises(RegistryError, match="PYTHONPATH"):
        get_model_class("not_a_package_xyz.mod:Net")


def test_load_weights_applies_convert_state_dict(tmp_path):
    cfg = load_config(None, ["model.params={num_layers: 1, dim_feedforward: 64}"])
    src = build_model(cfg, 1)
    original = {f"module.{k}": v for k, v in src.state_dict().items()}  # as saved by the original repo
    original["transformer_encoder_layer.linear1.weight"] = torch.zeros(1)
    torch.save(original, tmp_path / "orig.pt")
    dst = build_model(cfg, 1)
    load_weights(dst, tmp_path / "orig.pt")
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


# ----------------------------------------------------------------------------- the checker itself
class _Base(torch.nn.Module):
    def __init__(self, in_channels=2, in_samples=64, num_outputs=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, num_outputs, 1)


class _WrongShape(_Base):
    def forward(self, x):
        return self.conv(x)[:, :, None]


class _MixesBatch(_Base):
    def forward(self, x):
        return self.conv(x - x.mean(0, keepdim=True))


class _InPlace(_Base):
    def forward(self, x):
        x.mul_(2)
        return self.conv(x)


class _Detached(_Base):
    def forward(self, x):
        return self.conv(x).detach()


class _Unregistered(_Base):
    def __init__(self):
        super().__init__()
        self.scale = torch.rand(1)  # plain attribute: not saved in the state dict

    def forward(self, x):
        return self.conv(x) * self.scale


@pytest.mark.parametrize("cls, message", [
    (_WrongShape, "must have shape"), (_MixesBatch, "batch elements interact"), (_InPlace, "modified its input"),
    (_Detached, "no parameter received a gradient"), (_Unregistered, "fresh instance")])
def test_contract_checker_detects_violations(cls, message):
    with pytest.raises(ModelContractError, match=message):
        check_contract(cls, 2, 64, 1, batch_size=3)
