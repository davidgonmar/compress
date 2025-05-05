import unittest
from torch import nn
from compress.common import default_should_do, gather_submodules


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3)
        self.relu = nn.ReLU()
        self.seq = nn.Sequential(nn.Linear(10, 5), nn.BatchNorm1d(5))


class NestedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 1, kernel_size=1), nn.ReLU())
        self.layer2 = nn.ModuleList([nn.Linear(4, 2), nn.Linear(2, 1)])


class TestCompressCommon(unittest.TestCase):
    def setUp(self):
        self.model = DummyModule()

    def test_gather_submodules_default(self):
        result = gather_submodules(self.model, default_should_do)
        names = set(name for name, _ in result)
        expected_names = {"", "conv", "relu", "seq", "seq.0", "seq.1"}
        self.assertEqual(names, expected_names)

    def test_gather_submodules_custom_lambda(self):
        result = gather_submodules(
            self.model, lambda module, full_name: full_name.startswith("seq")
        )
        names = set(name for name, _ in result)
        expected_names = {"seq", "seq.0", "seq.1"}
        self.assertEqual(names, expected_names)

    def test_nested_module_structure(self):
        nested_model = NestedModule()
        result = gather_submodules(nested_model, default_should_do)
        names = set(name for name, _ in result)
        expected_names = {
            "",
            "layer1",
            "layer1.0",
            "layer1.1",
            "layer2",
            "layer2.0",
            "layer2.1",
        }
        self.assertEqual(names, expected_names)


if __name__ == "__main__":
    unittest.main()
