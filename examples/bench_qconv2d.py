import argparse
import time
import torch
import torch.nn as nn
from compress.quantization import IntQuantizationSpec, QuantizedConv2d


def benchmark_module(module, x, num_warmup=10, num_iters=1000, device="cpu"):
    module.eval()
    with torch.no_grad():
        # Warmup iterations.
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
    in_channels = args.in_channels
    out_channels = args.out_channels
    kernel_size = args.kernel_size
    height = args.height
    width = args.width
    x = torch.randn(batch_size, in_channels, height, width, device=device)

    fp32_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=True).to(
        device
    )
    fp32_conv2d.eval()

    qconv2d_simulated = QuantizedConv2d(
        weight_spec=weight_spec,
        input_info_or_spec=input_spec,
        conv2d=fp32_conv2d,
        simulated=True,
    ).to(device)

    qconv2d_non_simulated = QuantizedConv2d(
        weight_spec=weight_spec,
        input_info_or_spec=input_spec,
        conv2d=fp32_conv2d,
        simulated=False,
    ).to(device)

    # Compile the simulated quantized conv2d using torch.compile
    qconv2d_simulated_compiled = torch.compile(qconv2d_simulated)

    with torch.no_grad():
        out_fp32 = fp32_conv2d(x)
        out_q_sim = qconv2d_simulated(x)
        out_q_non_sim = qconv2d_non_simulated(x)

    err_sim = (out_fp32 - out_q_sim).abs().mean().item()
    err_non_sim = (out_fp32 - out_q_non_sim).abs().mean().item()
    print("Average absolute difference vs FP32 output:")
    print("  FP32 nn.Conv2d                : Reference")
    print(f"  Simulated QuantizedConv2d     : {err_sim:.4f}")
    print(f"  Non-simulated QuantizedConv2d : {err_non_sim:.4f}")

    num_iters = args.iters
    print(f"\nBenchmarking each module for {num_iters} iterations...")

    t_fp32 = benchmark_module(fp32_conv2d, x, num_iters=num_iters, device=device)
    t_sim = benchmark_module(qconv2d_simulated, x, num_iters=num_iters, device=device)
    t_non_sim = benchmark_module(
        qconv2d_non_simulated, x, num_iters=num_iters, device=device
    )
    t_sim_compiled = benchmark_module(
        qconv2d_simulated_compiled, x, num_iters=num_iters, device=device
    )

    print("\nAverage inference time per forward pass:")
    print(f"  FP32 nn.Conv2d                : {t_fp32:.3f} ms")
    print(f"  Simulated QuantizedConv2d     : {t_sim:.3f} ms")
    print(f"  Non-simulated QuantizedConv2d : {t_non_sim:.3f} ms")
    print(f"  Compiled Simulated QuantizedConv2d : {t_sim_compiled:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 nn.Conv2d vs. Simulated and Non-simulated QuantizedConv2d"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for input"
    )
    parser.add_argument(
        "--in-channels", type=int, default=4, help="Number of input channels"
    )
    parser.add_argument(
        "--out-channels", type=int, default=32, help="Number of output channels"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3, help="Kernel size for Conv2d"
    )
    parser.add_argument(
        "--height", type=int, default=32, help="Height of the input image"
    )
    parser.add_argument(
        "--width", type=int, default=32, help="Width of the input image"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of iterations for timing"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    args = parser.parse_args()
    main(args)
