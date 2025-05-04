#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.experiments import load_vision_model, get_cifar10_modifier

POOL_SIZE = 10
NUM_CLASSES = 10


class MINE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def make_hook(name):
    def hook(_, __, output):
        feat_buffers[name] = output.detach()

    return hook


def estimate_mi_vec(mine: MINE, z: torch.Tensor, y_onehot: torch.Tensor):
    z = F.adaptive_avg_pool2d(z, POOL_SIZE)
    z = z.flatten(1)

    joint = torch.cat([z, y_onehot], dim=1)
    y_perm = y_onehot[torch.randperm(z.size(0))]
    marginal = torch.cat([z, y_perm], dim=1)

    t_joint = mine(joint)
    t_marg = mine(marginal)
    return t_joint.mean(0) - torch.log(torch.exp(t_marg).mean(0) + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = load_vision_model(
        "resnet18",
        pretrained_path="resnet18.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet18"),
        modifier_after_load=None,
        model_args={"num_classes": NUM_CLASSES},
    ).to(device)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    global feat_buffers
    feat_buffers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.register_forward_hook(make_hook(name))

    mine_nets, mine_params = {}, []
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 32, 32, device=device))
        for name, z in feat_buffers.items():
            C = z.shape[1]
            in_dim = C * POOL_SIZE * POOL_SIZE + NUM_CLASSES
            mine = MINE(in_dim, C).to(device)
            mine_nets[name] = mine
            mine_params += list(mine.parameters())
            feat_buffers[name] = None

    mine_opt = optim.AdamW(mine_params, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        for net in mine_nets.values():
            net.train()

        epoch_mi_sum = {
            n: torch.zeros(mine_nets[n].fc2.out_features) for n in mine_nets
        }
        batches = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                _ = model(images)

            y_onehot = F.one_hot(labels, NUM_CLASSES).float()
            total_mi_scalar = 0.0

            for name, z in feat_buffers.items():
                mi_vec = estimate_mi_vec(mine_nets[name], z, y_onehot)
                epoch_mi_sum[name] += mi_vec.detach().cpu()
                total_mi_scalar += mi_vec.sum()

            batches += 1
            loss = -total_mi_scalar
            mine_opt.zero_grad()
            loss.backward()
            mine_opt.step()

        print(f"\nEpoch {epoch} — Mean MI per channel:")
        for name, mi_tot in epoch_mi_sum.items():
            ch_mean = (mi_tot / batches).numpy()
            preview = ", ".join(f"{v:.3f}" for v in ch_mean)
            print(f"{name}: {preview} … (C={len(ch_mean)})")

        if epoch % 10 == 0:
            for name, net in mine_nets.items():
                torch.save(net.state_dict(), f"mine_{name}.pt")


if __name__ == "__main__":
    main()
