import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from compress.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
    update_weights,
)
import torchvision


transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.resnet18(num_classes=10)
model.load_state_dict(torch.load("resnet18.pth"), strict=False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)


regularizer = SingularValuesRegularizer(
    metric="hoyer_sparsity",
    params_and_reshapers=extract_weights_and_reshapers(
        model,
        cls_list=(torch.nn.Linear, torch.nn.Conv2d),
        keywords={"weight", "kernel"},
    ),
    weights=1.0,
    normalize=True,
)


num_epochs = 350


def weight_schedule(epochnum):
    # penalty decreases exponentially from 3.0 to 0.1
    return 0.1 + 0.9 * (0.1 ** (epochnum / num_epochs)) * 3.0


for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0

    update_weights(regularizer, weight_schedule(epoch))

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

        torch.save(model, "cifar10_resnet18_hoyer_finetuned.pth")
