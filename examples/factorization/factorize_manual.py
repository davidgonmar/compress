import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorization.factorize import (
    to_low_rank_manual,
    all_same_svals_energy_ratio,
    all_same_params_ratio,
    all_same_rank_ratio,
)
from compress.flops import count_model_flops
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--load_from", type=str, default="resnet18.pth")
parser.add_argument("--vision-model", type=str, default="resnet18")
parser.add_argument("--keep_edge_layer", action="store_true")
parser.add_argument("--metric", type=str, default="energy")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_vision_model(
    args.vision_model,
    pretrained_path=args.load_from,
    strict=True,
    modifier_before_load=get_cifar10_modifier(args.vision_model),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

train_set = datasets.CIFAR10(
    root="data", train=True, transform=transform, download=True
)

subset_train_set = torch.utils.data.Subset(
    train_set, torch.randint(0, len(train_set), (10000,))
)
# only get a subset of train_loader
train_loader = DataLoader(subset_train_set, batch_size=100, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
eval_results = evaluate_vision_model(model, test_loader, eval=False)
n_params = sum(p.numel() for p in model.parameters())
flops = count_model_flops(model, (1, 3, 32, 32))
print(
    f"Test Loss: {eval_results['loss']}, Test Accuracy: {eval_results['accuracy']}, Number of parameters: {n_params}, Flops: {flops}"
)

energies = [
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9992,
    0.9995,
    0.9997,
    0.9999,
    0.99993,
    0.99995,
    0.99997,
    0.99999,
]

ratios = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

params_ratio = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]
for x in (
    energies
    if args.metric == "energy"
    else ratios if args.metric == "rank" else params_ratio
):
    cfg = (
        all_same_svals_energy_ratio(
            model,
            energy=x,
        )
        if args.metric == "energy"
        else (
            all_same_rank_ratio(
                model,
                rank=x,
            )
            if args.metric == "rank"
            else all_same_params_ratio(
                model,
                ratio=x,
            )
        )
    )
    if args.keep_edge_layer:
        del cfg["conv1"]
        del cfg["fc"]
    model_lr = to_low_rank_manual(
        model,
        cfg_dict=cfg,
        inplace=False,
    )
    n_params = sum(p.numel() for p in model_lr.parameters())
    model_lr.to(device)
    eval_results = evaluate_vision_model(model_lr, test_loader)

    fl = count_model_flops(model_lr, (1, 3, 32, 32))
    s = "Energy ratio" if args.metric == "energy" else "Rank ratio"
    print(
        f"{s}: {x:.8f}, Test Loss: {eval_results['loss']:.4f}, Test Accuracy: {eval_results['accuracy']:.4f}, Ratio of parameters: {n_params / sum(p.numel() for p in model.parameters()):.4f}, Flops: {fl}"
    )
