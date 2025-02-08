from compress.common import gather_submodules
import torch.nn as nn
import pytest


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.conv = nn.Conv2d(3, 6, kernel_size=3)
        self.seq = nn.Sequential(
            nn.Linear(5, 2),
            nn.Conv2d(6, 3, kernel_size=1),
            nn.Sequential(nn.Linear(2, 2)),
        )
        self.submodule = nn.Module()
        self.submodule.inner = nn.Linear(4, 2)

    def forward(self, x):
        return x


@pytest.fixture
def model():
    return DummyModel()


def test_gather_linear(model):
    def is_linear(module, name):
        return isinstance(module, nn.Linear)

    gathered = gather_submodules(model, is_linear)
    expected = {
        "fc": model.fc,
        "seq.0": model.seq[0],
        "seq.2.0": model.seq[2][0],
        "submodule.inner": model.submodule.inner,
    }
    gathered_dict = {name: mod for name, mod in gathered}
    assert gathered_dict == expected


def test_gather_conv(model):
    def is_conv(module, name):
        return isinstance(module, nn.Conv2d)

    gathered = gather_submodules(model, is_conv)
    expected = {"conv": model.conv, "seq.1": model.seq[1]}
    gathered_dict = {name: mod for name, mod in gathered}
    assert gathered_dict == expected


def test_gather_all(model):
    def always_true(module, name):
        return True

    gathered = gather_submodules(model, always_true)
    expected = {
        "fc": model.fc,
        "conv": model.conv,
        "seq.0": model.seq[0],
        "seq.1": model.seq[1],
        "seq.2": model.seq[2],
        "submodule.inner": model.submodule.inner,
    }
    gathered_dict = {name: mod for name, mod in gathered}
    assert gathered_dict == expected
