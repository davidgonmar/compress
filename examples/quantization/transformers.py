#!/usr/bin/env python
import warnings

warnings.filterwarnings("ignore")
import argparse
import time
import torch
from transformers import AutoModelForCausalLM
from compress.quantization import to_quantized_online, IntAffineQuantizationSpec


def main():
    parser = argparse.ArgumentParser(
        description="Llama‑2 text generation with optional quantization and tokens-per-second reporting."
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply quantization to the model (requires the compress.quantization package).",
    )
    args = parser.parse_args()
    model_name = "andrijdavid/Llama3-1B-Base"
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    if args.quantize:
        w8 = IntAffineQuantizationSpec(nbits=8, signed=True)
        a8 = IntAffineQuantizationSpec(nbits=8, signed=True)
        model = to_quantized_online(model, {"linear": w8}, {"linear": a8})
    model.eval()

    # batched generation
    input_ids = torch.randint(0, 50256, (16, 128), dtype=torch.long, device="cuda")

    # warmup
    with torch.no_grad():
        _ = model(input_ids)
        torch.cuda.synchronize()

    num_iters = 4
    start = time.time()
    for _ in range(num_iters):
        _ = model(input_ids)
    end = time.time()
    torch.cuda.synchronize()
    avg_s = (end - start) / num_iters

    print(f"Tokens per second: {128 * 128 / avg_s:.2f}")


if __name__ == "__main__":
    main()
