#!/usr/bin/env python3
import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot training run results and save to PDF"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to JSON result file")
    parser.add_argument(
        "-o", "--output", default="training_plot.pdf", help="Output PDF file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    with open(args.input) as f:
        runs = json.load(f)

    accuracies = [entry.get("accuracy") for entry in runs]
    epochs = [entry.get("epoch", idx + 1) for idx, entry in enumerate(runs)]

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracies, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved training plot to {args.output}")


if __name__ == "__main__":
    main()
