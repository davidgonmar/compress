from compress.factorization.factorize import plot_singular_values
import torch
import argparse
from compress.experiments import load_vision_model


parser = argparse.ArgumentParser(description="Plot singular values of a model")
parser.add_argument(
    "--pretrained_path", default="resnet20.pth", type=str, help="Path to the model"
)
parser.add_argument("--model", default="resnet20", type=str, help="Model name")

args = parser.parse_args()

model = load_vision_model(
    args.model,
    pretrained_path=args.pretrained_path,
    strict=True,
    modifier_before_load=None,
    modifier_after_load=None,
    model_args={"num_classes": 10},
    accept_model_directly=True,
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

plot_singular_values(model)
