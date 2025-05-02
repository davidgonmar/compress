import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from compress.quantization import prepare_for_qat
from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.quantization.recipes import get_resnet18_recipe_quant


class MINE(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


feat_buffers = {}


def make_hook(name):
    def hook(_, __, output):
        feat_buffers[name] = output

    return hook


def estimate_mi(mine, z, y_onehot):
    b = z.size(0)
    z = z.mean(dim=(2, 3))
    joint = torch.cat([z, y_onehot], dim=1)
    y_perm = y_onehot[torch.randperm(b)]
    marginal = torch.cat([z, y_perm], dim=1)
    t_joint = mine(joint)
    t_marg = mine(marginal)
    return t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)


parser = argparse.ArgumentParser()
parser.add_argument("--method", default="qat", type=str)
parser.add_argument("--nbits", default=2, type=int)
parser.add_argument(
    "--leave_last_layer_8_bits", type=lambda x: str(x).lower() == "true", default=True
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=data_transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=data_transform
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

model = load_vision_model(
    "resnet18",
    pretrained_path="resnet18.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)
specs = get_resnet18_recipe_quant(
    bits_activation=args.nbits,
    bits_weight=args.nbits,
    leave_edge_layers_8_bits=args.leave_last_layer_8_bits,
    clip_percentile=0.99,
    symmetric=True,
)
model = prepare_for_qat(
    model,
    specs=specs,
    use_lsq=True,
    use_PACT=True,
    data_batch=next(iter(train_loader))[0][:4].to(device),
).to(device)

for n, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        m.register_forward_hook(make_hook(n))

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

mine_nets, mine_params, mine_opt = {}, [], None
NUM_CLASSES, lambda_MI = 10, 0.01

for epoch in range(100):
    model.train()
    running_ce, running_mi = 0.0, 0.0
    for step, (images, labels) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False)
    ):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        ce_loss = criterion(logits, labels)
        if epoch == 0 and step == 0:
            for name, z in feat_buffers.items():
                mine = MINE(z.shape[1] + NUM_CLASSES).to(device)
                mine_nets[name] = mine
                mine_params += list(mine.parameters())
            mine_opt = optim.AdamW(mine_params, lr=1e-4)
        y_onehot = F.one_hot(labels, NUM_CLASSES).float()
        total_mi = sum(
            estimate_mi(mine_nets[name], z, y_onehot)
            for name, z in feat_buffers.items()
        )
        loss = ce_loss - lambda_MI * total_mi
        optimizer.zero_grad()
        mine_opt.zero_grad()
        loss.backward()
        optimizer.step()
        mine_opt.step()
        running_ce += ce_loss.item() * images.size(0)
        running_mi += total_mi.item() * images.size(0)
    scheduler.step()
    print(
        f"Epoch {epoch+1}, CE {running_ce/len(train_loader.dataset):.4f}, MI {running_mi/len(train_loader.dataset):.4f}"
    )
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1} - Validation", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")
