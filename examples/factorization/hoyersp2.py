import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.factorization.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
)
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
)
from compress.factorization.utils import matrix_approx_rank

transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

train_dataset = datasets.CIFAR10(
    root="data", train=True, download=True, transform=transform
)
val_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    ),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_vision_model(
    "resnet20",
    pretrained_path="resnet20.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)


regularizer = SingularValuesRegularizer(
    metric="hoyer_sparsity",
    params_and_reshapers=extract_weights_and_reshapers(
        model,
        cls_list=(torch.nn.Linear, torch.nn.Conv2d),
        keywords={"weight", "kernel"},
    ),
    weights=1.0,
    normalize=False,
)


num_epochs = 100

START = 0.01
END = 0.001
import math


def weight_schedule(ep, T_0=20, T_mult=1):
    T_i = T_0
    ep_i = ep
    # determine current cycle
    while ep_i >= T_i:
        ep_i -= T_i
        T_i *= T_mult
    # cosine annealing within the cycle
    return END + 0.5 * (START - END) * (1 + math.cos(math.pi * ep_i / T_i))


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
        weight = weight_schedule(epoch)
        reg = regularizer()
        total_loss = loss + weight * reg

        total_loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        reg_loss += reg.item() * x.size(0)

    train_loss /= len(train_loader.dataset)
    reg_loss /= len(train_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Regularization Loss: {reg_loss:.4f}"
    )

    def mul(shape):
        result = 1
        for dim in shape:
            result *= dim
        return result

    print("approx ranks:")
    for name, param in model.named_modules():
        if isinstance(param, torch.nn.Conv2d):
            approx_rank = matrix_approx_rank(param.weight)
            print(
                f"{name}: {approx_rank} out of {param.weight.shape[0]}, {mul(param.weight.shape[1:])} total"
            )
        elif isinstance(param, torch.nn.Linear):
            approx_rank = matrix_approx_rank(param.weight)
            print(
                f"{name}: {approx_rank} out of {param.weight.shape[0]}, {mul(param.weight.shape[1:])} total"
            )
    if epoch % 5 == 0 or True:
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
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

        torch.save(model, "cifar10_resnet18_hoyer_finetuned.pth")
