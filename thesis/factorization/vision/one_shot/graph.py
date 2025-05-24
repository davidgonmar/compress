#!/usr/bin/env python
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_curve(json_path, x_key, y_key):
    with Path(json_path).open() as f:
        data = json.load(f)
    return [item[x_key] for item in data], [item[y_key] for item in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_flops_json", required=True)
    parser.add_argument("--global_params_json", required=True)
    parser.add_argument("--global_activation_aware_flops_json", required=True)
    parser.add_argument("--global_activation_aware_params_json", required=True)

    parser.add_argument("--manual_rank_json", required=True)
    parser.add_argument("--manual_energy_json", required=True)
    parser.add_argument("--manual_activation_aware_rank_json", required=True)
    parser.add_argument("--manual_activation_aware_energy_json", required=True)
    parser.add_argument("--output_flops", default="flops_vs_acc.pdf")
    parser.add_argument("--output_params", default="params_vs_acc.pdf")
    args = parser.parse_args()

    def plot_flops_vs_acc():

        # global
        xs, ys = load_curve(args.global_flops_json, "flops_ratio", "accuracy")
        plt.plot(xs, ys, label="Global (FLOPs)")

        xs, ys = load_curve(
            args.global_activation_aware_flops_json, "flops_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Global Activation-aware (FLOPs)")

        # manual uniform
        xs, ys = load_curve(args.manual_rank_json, "flops_ratio", "accuracy")
        plt.plot(xs, ys, label="Uniform (Rank)")

        xs, ys = load_curve(
            args.manual_activation_aware_rank_json, "flops_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Uniform Activation-aware (Rank)")

        # manual energy
        xs, ys = load_curve(args.manual_energy_json, "flops_ratio", "accuracy")
        plt.plot(xs, ys, label="Uniform (Energy)")

        xs, ys = load_curve(
            args.manual_activation_aware_energy_json, "flops_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Uniform Activation-aware (Energy)")

        plt.xlabel("FLOPs (Millions)")
        plt.ylabel("Top-1 Accuracy (%)")

        plt.legend()
        plt.savefig(args.output_flops)
        plt.close()

    def plot_params_vs_acc():
        # global
        xs, ys = load_curve(args.global_params_json, "params_ratio", "accuracy")
        plt.plot(xs, ys, label="Global (Params)")

        xs, ys = load_curve(
            args.global_activation_aware_params_json, "params_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Global Activation-aware (Params)")

        # manual uniform
        xs, ys = load_curve(args.manual_rank_json, "params_ratio", "accuracy")
        plt.plot(xs, ys, label="Uniform (Rank)")

        xs, ys = load_curve(
            args.manual_activation_aware_rank_json, "params_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Uniform Activation-aware (Rank)")

        # manual energy
        xs, ys = load_curve(args.manual_energy_json, "params_ratio", "accuracy")
        plt.plot(xs, ys, label="Uniform (Energy)")

        xs, ys = load_curve(
            args.manual_activation_aware_energy_json, "params_ratio", "accuracy"
        )
        plt.plot(xs, ys, label="Uniform Activation-aware (Energy)")

        plt.xlabel("Params (Millions)")
        plt.ylabel("Top-1 Accuracy (%)")

        plt.legend()
        plt.savefig(args.output_params)
        plt.close()

    plot_flops_vs_acc()
    plot_params_vs_acc()


if __name__ == "__main__":
    main()
