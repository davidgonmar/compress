import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
    update_weights,
)
import argparse
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torchvision
from tqdm import tqdm
from compress.factorize import to_low_rank, merge_back

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--load_path", type=str, default="None")
parser.add_argument("--sv_regularizer", type=str, default="noop")
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--regularizer_weight", type=float, default=1.0)
parser.add_argument("--regularizer_scheduler", type=str, default="noop")
parser.add_argument("--factorize_each", type=int, default=3)
parser.add_argument("--energy", type=float, default=0.99999)
args = parser.parse_args()


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=transform, download=True
    )
    val_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    return train_loader, val_loader


train_loader, val_loader = get_cifar10()
criterion = torch.nn.CrossEntropyLoss()

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 10)

if args.load_path != "None":
    _model = torch.load(args.load_path, weights_only=False)
    if isinstance(_model, torch.nn.Module):
        model.load_state_dict(_model.state_dict())
    else:
        model.load_state_dict(_model)


optimizer = optim.AdamW(model.parameters(), lr=0.001)
sched = StepLR(optimizer, step_size=10, gamma=0.2)
# load weights from pre-trained model from torch


regularizer_kwargs = {
    "entropy": {},
    "hoyer_sparsity": {"normalize": True},
    "scad": {"lambda_val": 0.1, "a_val": 3.7},
    "squared_hoyer_sparsity": {"normalize": True},
    "nuclear_norm": {},
    "approximated_hoyer_sparsity": {"normalize": True},
    "noop": {},
}

weights = args.regularizer_weight


def weight_schedule_noop(epochnum):
    return weights


def weight_schedule_exp(epochnum):
    # penalty is gradually more important
    return 0.5 * weights * 2 ** ((epochnum / args.epochs))


weight_sched = {
    "noop": weight_schedule_noop,
    "exp": weight_schedule_exp,
}[args.regularizer_scheduler]

nofparams = sum(p.numel() for p in model.parameters())
regularizer = SingularValuesRegularizer(
    metric=args.sv_regularizer,
    params_and_reshapers=extract_weights_and_reshapers(
        model,
        cls_list=(
            torch.nn.Linear,
            torch.nn.Conv2d,
            torch.nn.LazyLinear,
            torch.nn.LazyConv2d,
        ),
        keywords={"weight", "kernel"},
    ),
    weights=args.regularizer_weight,
    **regularizer_kwargs[args.sv_regularizer],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0
    update_weights(regularizer, weight_sched(epoch))
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        reg = regularizer()
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
        reg_loss += reg.item() * x.size(0)

    train_loss /= len(train_loader.dataset)
    reg_loss /= len(train_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Regularization Loss: {reg_loss:.4f}"
    )

    if epoch % args.factorize_each == 0:
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
        print(
            f"BEFORE FACTORIZE: Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        model = to_low_rank(model, inplace=True, energy_to_keep=args.energy)
        nofparams2 = sum(p.numel() for p in model.parameters())
        model = merge_back(model, inplace=True)

        # print(model)
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
        print(
            f"AFTER FACTORIZE: Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}. Percentage of parameters kept: {nofparams2/nofparams:.4f}"
        )
        torch.save(model, args.save_path)


print("Finished training. Saving model...")
torch.save(model, args.save_path)
