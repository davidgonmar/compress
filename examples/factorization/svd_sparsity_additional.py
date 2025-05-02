import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torchvision
from tqdm import tqdm
from compress.factorization.factorize import hoyer_svd_sparsity_grad_adder
from compress.regularizers import singular_values_hoyer_sparsity

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="cifar_resnet18_model.pth")
parser.add_argument("--sv_regularizer", type=str, default="noop")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--regularizer_weight", type=float, default=1.0)
parser.add_argument("--regularizer_scheduler", type=str, default="noop")
parser.add_argument("--finetune", action="store_true")
args = parser.parse_args()


def get_cifar10():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=transform_train, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="data", train=False, transform=transform_test, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    return train_loader, test_loader, {"input_size": (3, 32, 32), "num_classes": 10}


train_loader, val_loader, model_params = get_cifar10()

model = resnet18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
sched = StepLR(optimizer, step_size=10, gamma=0.2)

if args.finetune:
    torch_weights = torchvision.models.resnet18(pretrained=True).state_dict()
    del torch_weights["fc.weight"]
    del torch_weights["fc.bias"]
    model.load_state_dict(torch_weights, strict=False)

weights = args.regularizer_weight


def weight_schedule_noop(epochnum):
    return weights


def weight_schedule_exp(epochnum):
    return 0.5 * weights * 2 ** (epochnum / args.epochs)


weight_sched = {
    "noop": weight_schedule_noop,
    "exp": weight_schedule_exp,
}[args.regularizer_scheduler]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        total_loss = loss
        total_loss.backward()
        hoyer_svd_sparsity_grad_adder(model.parameters(), weight_sched(epoch))
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    # measure hoyer sparsity of svd vals
    li = []
    for param in model.parameters():
        if param.dim() not in [2, 4]:
            continue
        param_frob = param.reshape(-1).pow(2).sum().sqrt()
        if param.dim() == 4:
            # reshape to (O, I * H * W) from (O, I, H, W)
            param_rs = param.reshape(param.shape[0], -1)
        else:
            param_rs = param
        r = singular_values_hoyer_sparsity(param_rs, True)
        li.append(r.item())

    print(
        f"Epoch {epoch+1}/{args.epochs}, Hoyer Sparsity: {[f'{val:.6f}' for val in li]}"
    )
    if epoch % 5 == 0:
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
        torch.save(model, args.save_path)

print("Finished training. Saving model...")
torch.save(model, args.save_path)
