import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path

from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.sparsity.prune import to_pruned_mask_provided, PruningPolicy


class MINE(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


parser = argparse.ArgumentParser()
parser.add_argument("--mine_ckpt_dir", default="./", type=str)
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--prune_percent", type=float, default=0.1)
parser.add_argument("--output", default="global_channel_mask.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True, transform=transform),
    batch_size=args.batch_size,
    shuffle=False,
)
NUM_CLASSES = 10

model = load_vision_model(
    "resnet18",
    pretrained_path="resnet18.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    modifier_after_load=None,
    model_args={"num_classes": NUM_CLASSES},
).to(device)
model.eval()

feat_buffers = {}


def _save_conv_out(_, __, out, layer_name):
    feat_buffers[layer_name] = out.detach()


for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        m.register_forward_hook(
            lambda *args, layer_name=name, fn=_save_conv_out: fn(*args, layer_name)
        )

mine_nets = {}
with torch.no_grad():
    _ = model(torch.randn(1, 3, 32, 32, device=device))
    for name, z in feat_buffers.items():
        C = z.shape[1]
        mine = MINE(C + NUM_CLASSES, C).to(device)
        ckpt_path = Path(args.mine_ckpt_dir) / f"mine_{name}.pt"
        mine.load_state_dict(torch.load(ckpt_path, map_location=device))
        mine.eval()
        mine_nets[name] = mine

feat_buffers.clear()


@torch.no_grad()
def mi_vector(net: MINE, z: torch.Tensor, y_onehot: torch.Tensor):
    z = z.mean(dim=(2, 3))
    joint = torch.cat([z, y_onehot], 1)
    y_perm = y_onehot[torch.randperm(z.size(0))]
    marginal = torch.cat([z, y_perm], 1)

    t_joint = net(joint)
    t_marg = net(marginal)

    return (t_joint.mean(0) - torch.log(torch.exp(t_marg).mean(0) + 1e-8)).cpu()


mi_scores = {name: torch.zeros(net.fc2.out_features) for name, net in mine_nets.items()}
counts = {name: 0 for name in mine_nets}

for imgs, lbls in tqdm(loader, desc="Estimating MI"):
    imgs, lbls = imgs.to(device), lbls.to(device)

    with torch.no_grad():
        _ = model(imgs)

    y_one = F.one_hot(lbls, NUM_CLASSES).float()

    for name, z in feat_buffers.items():
        mi_scores[name] += mi_vector(mine_nets[name], z, y_one)
        counts[name] += 1

    feat_buffers.clear()

for name in mi_scores:
    mi_scores[name] /= counts[name]

all_mi = torch.cat([v for v in mi_scores.values()])
threshold = torch.quantile(all_mi, args.prune_percent)

mask_dict = {}
for name, vec in mi_scores.items():
    mask = (vec > threshold).float().view(-1, 1, 1, 1)
    mask_dict[name] = mask

pruned_model = to_pruned_mask_provided(
    model,
    mask_dict,
    PruningPolicy(),
    should_do=lambda m, n: isinstance(m, nn.Conv2d),
    inplace=False,
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=transform),
    batch_size=args.batch_size,
    shuffle=False,
)
criterion = nn.CrossEntropyLoss()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        logits = pruned_model(images)
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the pruned model: {100 * correct / total:.2f}%")
