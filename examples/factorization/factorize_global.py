import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorization.factorize import to_low_rank_global, to_low_rank, to_low_rank_global2
from compress.flops import count_model_flops
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--load_from", type=str, default="mnist_model.pth")
parser.add_argument("--keep_edge_layer", action="store_true")
parser.add_argument("--do_global", action="store_true")
parser.add_argument("--do_global2", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_vision_model(
    "resnet18",
    pretrained_path=args.load_from,
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

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


def should_do(module, name):
    cond1 = isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))
    cond2 = True if not args.keep_edge_layer else (name not in ["conv1", "fc"])
    return cond1 and cond2


import functools

fn = None
if args.do_global:
    fn = to_low_rank_global
elif args.do_global2:
    fn = functools.partial(to_low_rank_global2, dataloader=train_loader)
else:
    fn = to_low_rank

energies = [
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
for ratio in energies:
    model_lr = fn(
        model,
        energy_to_keep=ratio,
        inplace=False,
        should_do=should_do,
    )
    n_params = sum(p.numel() for p in model_lr.parameters())
    model_lr.to(device)
    eval_results = evaluate_vision_model(model_lr, test_loader)

    fl = count_model_flops(model_lr, (1, 3, 32, 32))
    print(
        f"Ratio: {ratio:.8f}, Test Loss: {eval_results['loss']:.4f}, Test Accuracy: {eval_results['accuracy']:.4f}, Ratio of parameters: {n_params / sum(p.numel() for p in model.parameters()):.4f}"
    )
