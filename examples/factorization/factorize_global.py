import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.factorization.factorize import (
    to_low_rank_global,
)
from compress.flops import count_model_flops
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean,
    cifar10_std,
)
import argparse
from compress.utils import get_all_convs_and_linears


parser = argparse.ArgumentParser()
parser.add_argument("--keep_edge_layer", action="store_true")
parser.add_argument("--do_global", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_vision_model(
    "resnet20",
    pretrained_path="cifar10_resnet20_hoyer_finetuned.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)],
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
eval_results = evaluate_vision_model(model, test_loader)
n_params = sum(p.numel() for p in model.parameters())
flops = count_model_flops(model, (1, 3, 32, 32))
print(
    f"Test Loss: {eval_results['loss']}, Test Accuracy: {eval_results['accuracy']}, Number of parameters: {n_params}, Flops: {flops}"
)


keys = get_all_convs_and_linears(model)
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
    0.95,
]
for ratio in ratios:
    model_lr = to_low_rank_global(
        model,
        input_shape=(1, 3, 32, 32),
        ratio_to_keep=ratio,
        inplace=False,
        keys=keys,
        metric="flops",
    )
    n_params = sum(p.numel() for p in model_lr.parameters())
    model_lr.to(device)
    eval_results = evaluate_vision_model(model_lr, test_loader)

    fl = count_model_flops(model_lr, (1, 3, 32, 32))
    print(
        f"Ratio: {ratio:.8f}, Test Loss: {eval_results['loss']:.4f}, Test Accuracy: {eval_results['accuracy']:.4f}, Ratio of parameters: {n_params / sum(p.numel() for p in model.parameters()):.4f}, Flops: {fl}"
    )

    torch.save(model_lr, f"resnet20_lr_{ratio}.pth")
