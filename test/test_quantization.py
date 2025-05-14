import unittest
import torch
import torch.nn as nn
from compress.quantization.common import (
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
)
from compress.quantization import (
    get_quant_dict,
    to_quantized_online,
    to_quantized_offline,
    prepare_for_qat,
    merge_qat_model,
    merge_qat_lsq_into_offline_quantized_model,
    to_quantized_kmeans,
)
from compress.quantization.qat import (
    QATConv2d,
    QATLinear,
    LSQConv2d,
    LSQLinear,
    FusedQATConv2dBatchNorm2d,
    FusedLSQConv2dBatchNorm2d,
)
from compress.quantization.ptq import QuantizedLinear, QuantizedConv2d
from compress.quantization import KMeansQuantizedLinear, KMeansQuantizedConv2d


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(2)
        self.conv = nn.Conv2d(3, 8, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=[2, 3])
        return self.linear(x)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        # specs per layer type
        self.specs = {
            "linear": {
                "input_spec": IntAffineQuantizationSpec(
                    8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
                ),
                "weight_spec": IntAffineQuantizationSpec(
                    8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
                ),
            },
            "conv": {
                "input_spec": IntAffineQuantizationSpec(
                    8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
                ),
                "weight_spec": IntAffineQuantizationSpec(
                    8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
                ),
            },
        }
        self.data = torch.randn(2, 3, 8, 8)
        # initialize lazy layers
        self.model(self.data)

    def _build_specs_dict(self):
        # helper: gather specs for both linear and conv layers
        specs = get_quant_dict(
            self.model,
            "linear",
            input_spec=self.specs["linear"]["input_spec"],
            weight_spec=self.specs["linear"]["weight_spec"],
        )
        specs.update(
            get_quant_dict(
                self.model,
                "conv2d",  # conv layers
                input_spec=self.specs["conv"]["input_spec"],
                weight_spec=self.specs["conv"]["weight_spec"],
            )
        )
        return specs

    def test_to_quantized_online(self):
        specs = self._build_specs_dict()
        quant_model = to_quantized_online(self.model, specs, inplace=False)
        self.assertIsInstance(quant_model.linear, QuantizedLinear)
        self.assertIsInstance(quant_model.conv, QuantizedConv2d)

    def test_to_quantized_offline(self):
        specs = self._build_specs_dict()
        quant_model = to_quantized_offline(
            self.model, specs, data_loader=self.data, inplace=False
        )
        self.assertIsInstance(quant_model.linear, QuantizedLinear)
        self.assertIsInstance(quant_model.conv, QuantizedConv2d)

    def test_prepare_for_qat(self):
        specs = self._build_specs_dict()
        qat_model = prepare_for_qat(self.model, specs, inplace=False)
        self.assertIsInstance(qat_model.linear, QATLinear)
        self.assertIsInstance(qat_model.conv, QATConv2d)

    def test_prepare_for_lsq_qat(self):
        specs = self._build_specs_dict()
        qat_model = prepare_for_qat(
            self.model,
            specs,
            use_lsq=True,
            data_batch=self.data,
            inplace=False,
        )
        self.assertIsInstance(qat_model.linear, LSQLinear)
        self.assertIsInstance(qat_model.conv, LSQConv2d)

    @unittest.skip("broken")
    def test_merge_qat_model(self):
        specs = self._build_specs_dict()
        qat_model = prepare_for_qat(self.model, specs, inplace=False)
        merged = merge_qat_model(qat_model, inplace=False)
        self.assertIsInstance(merged.linear, nn.Linear)
        self.assertIsInstance(merged.conv, nn.Conv2d)

    @unittest.skip("broken")
    def test_merge_lsq_into_offline(self):
        specs = self._build_specs_dict()
        qat_model = prepare_for_qat(
            self.model,
            specs,
            use_lsq=True,
            data_batch=self.data,
            inplace=False,
        )
        merged = merge_qat_lsq_into_offline_quantized_model(qat_model, inplace=False)
        self.assertTrue(hasattr(merged.linear, "forward"))
        self.assertTrue(hasattr(merged.conv, "forward"))

    @unittest.skip("KMeansQuantization broken")
    def test_to_quantized_kmeans(self):
        kmeans_model = to_quantized_kmeans(
            self.model,
            input_specs={
                "linear": self.specs["linear"]["input_spec"],
                "conv2d": self.specs["conv"]["input_spec"],
            },
            weight_specs={
                "linear": self.specs["linear"]["weight_spec"],
                "conv2d": self.specs["conv"]["weight_spec"],
            },
            inplace=False,
        )
        self.assertIsInstance(kmeans_model.linear, KMeansQuantizedLinear)
        self.assertIsInstance(kmeans_model.conv, KMeansQuantizedConv2d)


class SimpleModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(2)
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = x.mean(dim=[2, 3])
        return self.linear(x)


class TestBatchNormFusion(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel2()
        self.data = torch.randn(2, 3, 8, 8)
        # initialize lazy layers
        self.model(self.data)

        self.fuse_keys = [
            ("conv", "bn"),
        ]

        self.specs = {
            "input_spec": IntAffineQuantizationSpec(
                8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
            ),
            "weight_spec": IntAffineQuantizationSpec(
                8, signed=True, quant_mode=IntAffineQuantizationMode.SYMMETRIC
            ),
        }

    def _build_specs_dict(self):
        # helper: gather specs for both linear and conv layers
        specs = get_quant_dict(
            self.model,
            "linear",
            input_spec=self.specs["input_spec"],
            weight_spec=self.specs["weight_spec"],
        )
        specs.update(
            get_quant_dict(
                self.model,
                "conv2d",  # conv layers
                input_spec=self.specs["input_spec"],
                weight_spec=self.specs["weight_spec"],
            )
        )
        return specs

    def test_fuse_conv_bn_qat(self):
        specs = self._build_specs_dict()
        qat = prepare_for_qat(
            self.model, inplace=False, specs=specs, fuse_bn_keys=self.fuse_keys
        )
        # assert conv and bn layers are fused
        self.assertIsInstance(qat.conv, FusedQATConv2dBatchNorm2d)
        self.assertIsInstance(qat.bn, nn.Identity)

        self.assertIsInstance(qat.linear, QATLinear)

    def test_fuse_conv_bn_lsq(self):
        specs = self._build_specs_dict()
        qat = prepare_for_qat(
            self.model,
            inplace=False,
            specs=specs,
            use_lsq=True,
            fuse_bn_keys=self.fuse_keys,
            data_batch=self.data,
        )
        # assert conv and bn layers are fused
        self.assertIsInstance(qat.conv, FusedLSQConv2dBatchNorm2d)
        self.assertIsInstance(qat.bn, nn.Identity)

        self.assertIsInstance(qat.linear, LSQLinear)

    def test_fuse_conv_bn_offline(self):
        specs = self._build_specs_dict()
        qat = to_quantized_offline(
            self.model,
            inplace=False,
            specs=specs,
            data_loader=self.data,
            fuse_bn_keys=self.fuse_keys,
        )
        # assert conv and bn layers are fused
        self.assertIsInstance(qat.conv, QuantizedConv2d)
        self.assertIsInstance(qat.bn, nn.Identity)

        self.assertIsInstance(qat.linear, QuantizedLinear)


if __name__ == "__main__":
    unittest.main()
