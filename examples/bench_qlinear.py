import argparse
import time
import torch
import torch.nn as nn
from compress.quantization import IntQuantizationSpec, QuantizedLinear


def benchmark_module(module, x, num_warmup=10, num_iters=1000, device="cpu"):
    module.eval()
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = module(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            _ = module(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        end = time.time()
    avg_ms = (end - start) / num_iters * 1000
    return avg_ms


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    weight_spec = IntQuantizationSpec(nbits=8, signed=True)
    input_spec = IntQuantizationSpec(nbits=8, signed=True)

    batch_size = args.batch_size
    in_features = args.in_features
    out_features = args.out_features
    x = torch.randn(batch_size, in_features, device=device)

    fp32_linear = nn.Linear(in_features, out_features, bias=True).to(device)
    fp32_linear.eval()

    qlinear_simulated = QuantizedLinear(
        weight_spec=weight_spec,
        input_info_or_spec=input_spec,
        linear=fp32_linear,
        simulated=True,
    ).to(device)

    qlinear_non_simulated = QuantizedLinear(
        weight_spec=weight_spec,
        input_info_or_spec=input_spec,
        linear=fp32_linear,
        simulated=False,
    ).to(device)

    with torch.no_grad():
        out_fp32 = fp32_linear(x)
        out_q_sim = qlinear_simulated(x)
        out_q_non_sim = qlinear_non_simulated(x)

    err_sim = (out_fp32 - out_q_sim).abs().mean().item()
    err_non_sim = (out_fp32 - out_q_non_sim).abs().mean().item()
    print("Average absolute difference vs FP32 output:")
    print("  FP32 nn.Linear                : Reference")
    print(f"  Simulated QuantizedLinear     : {err_sim:.4f}")
    print(f"  Non-simulated QuantizedLinear : {err_non_sim:.4f}")

    num_iters = args.iters
    print(f"\nBenchmarking each module for {num_iters} iterations...")

    t_fp32 = benchmark_module(fp32_linear, x, num_iters=num_iters, device=device)
    t_sim = benchmark_module(qlinear_simulated, x, num_iters=num_iters, device=device)
    t_non_sim = benchmark_module(
        qlinear_non_simulated, x, num_iters=num_iters, device=device
    )

    print("\nAverage inference time per forward pass:")
    print(f"  FP32 nn.Linear                : {t_fp32:.3f} ms")
    print(f"  Simulated QuantizedLinear     : {t_sim:.3f} ms")
    print(f"  Non-simulated QuantizedLinear : {t_non_sim:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 nn.Linear vs. Simulated and Non-simulated QuantizedLinear"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for input"
    )
    parser.add_argument(
        "--in-features", type=int, default=2048, help="Number of input features"
    )
    parser.add_argument(
        "--out-features", type=int, default=4096, help="Number of output features"
    )
    parser.add_argument(
        "--iters", type=int, default=1000, help="Number of iterations for timing"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    args = parser.parse_args()
    main(args)
