# Similar to https://arxiv.org/abs/2004.09031
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from compress.factorization.factorize import to_low_rank
from compress.regularizers import OrthogonalRegularizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        RandomHorizontalFlip(),
        RandomCrop(128, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=data_transform
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=data_transform
)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
model = resnet18(num_classes=10).to(device)
model = to_low_rank(
    model, inplace=True, ratio_to_keep=1.0, keep_singular_values_separated=True
)  # just factorize, do not compress
criterion = nn.CrossEntropyLoss()
reg = OrthogonalRegularizer.apply_to_low_rank_modules(
    model, weights=8.0, normalize_by_rank_squared=True
)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

for epoch in range(30):
    model.train()
    train_loss_acc = 0.0
    reg_loss_acc = 0.0
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch + 1} - Training", leave=False
    ):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        reg_loss = reg()
        (train_loss + reg_loss).backward()
        optimizer.step()

        train_loss_acc += train_loss.item() * images.size(0)
        reg_loss_acc += reg_loss.item() * images.size(0)

    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Loss: {train_loss_acc / len(train_loader.dataset):.4f}, Regularizer Loss: {reg_loss_acc / len(train_loader.dataset):.4f}"
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch + 1} - Validation", leave=False
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    reg_loss = reg()
    print(
        f"Epoch {epoch + 1}, Accuracy: {accuracy:.2f}%, Regularizer Loss: {reg_loss.item():.4f}"
    )
    torch.save(model, "low_rank_resnet.pth")
