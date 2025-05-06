import torch

import argparse
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from compress.experiments import load_vision_model, get_cifar10_modifier
from compress.sparsity.regularizers import (
    SparsityActivationRegularizer,
    get_regularizer_for_all_layers,
    L1L2ActivationInterRegularizer,
)
from compress.sparsity.pruning_strats import (
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
)
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--save_path", type=str, default="sparsity_model.pth")
args = parser.parse_args()


model = load_vision_model(
    model_str="resnet18",
    pretrained_path="resnet18.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet18"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=0.001)
sched = StepLR(optimizer, step_size=10, gamma=0.2)


regularizer = SparsityActivationRegularizer(
    model,
    get_regularizer_for_all_layers(
        model,
        regfn=L1L2ActivationInterRegularizer(),
        conv_grouper=OutChannelGroupingGrouperConv2d(),
        linear_grouper=OutChannelGroupingGrouperLinear(),
    ),
)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=True, download=True, transform=transforms),
    batch_size=256,
    shuffle=True,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("./data", train=False, download=True, transform=transforms),
    batch_size=512,
    shuffle=False,
    pin_memory=True,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        reg = regularizer.loss()
        total_loss = loss + 0.1 * reg

        total_loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        reg_loss += reg.item() * x.size(0)

    train_loss /= len(train_loader.dataset)
    reg_loss /= len(train_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Regularization Loss: {reg_loss:.4f}"
    )

    if epoch % 2 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"
            ):
                x, y = batch
                x, y = x.to(device), y.to(device)

                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_loss += loss.item() * x.size(0)

                preds = y_hat.argmax(dim=1)
                correct += (preds == y).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        import copy

        def remove_all_hooks(module: torch.nn.Module):
            # these are OrderedDicts mapping hook_id → hook_fn
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            if hasattr(module, "_forward_pre_hooks"):
                module._forward_pre_hooks.clear()
            if hasattr(module, "_backward_hooks"):
                module._backward_hooks.clear()

        model_copy = copy.deepcopy(model)

        model_copy.apply(remove_all_hooks)

        torch.save(model_copy, args.save_path)
