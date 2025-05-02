from compress.factorization.factorize import plot_singular_values
import torch
import argparse


parser = argparse.ArgumentParser(description="Plot singular values of a model")
parser.add_argument("--model", type=str, help="Path to the model")
args = parser.parse_args()

model = torch.load(args.model, weights_only=False)
plot_singular_values(model)
