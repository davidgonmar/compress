import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean,
    cifar10_std,
)

torch.manual_seed(0)


def get_data_loaders(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return trainloader, testloader


class StudentConv(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(6 * 14 * 14, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.9):
    soft_targets = nn.functional.log_softmax(student_logits / T, dim=1)
    soft_labels = nn.functional.softmax(teacher_logits / T, dim=1)
    loss_kd = nn.functional.kl_div(soft_targets, soft_labels, reduction="batchmean") * (
        T * T
    )
    loss_ce = nn.functional.cross_entropy(student_logits, targets)
    return alpha * loss_kd + (1.0 - alpha) * loss_ce


def train_student(
    student, teacher, train_loader, optimizer, device, use_kd=False, T=2.0, alpha=0.9
):
    student.train()
    if use_kd:
        teacher.eval()
    running_loss = 0.0
    correct = total = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if use_kd:
            with torch.no_grad():
                t_logits = teacher(inputs)
            s_logits = student(inputs)
            loss = distillation_loss(s_logits, t_logits, targets, T, alpha)
        else:
            s_logits = student(inputs)
            loss = nn.functional.cross_entropy(s_logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = s_logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss / len(train_loader.dataset), 100.0 * correct / total


def evaluate_metrics(model, test_loader, device):
    ret = evaluate_vision_model(model, test_loader)
    return ret["accuracy"], ret["loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", default="runs/kd_vs_ce", help="TensorBoard log directory"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders(batch_size=128)
    writer = SummaryWriter(log_dir=args.logdir)
    teacher = load_vision_model(
        "resnet18",
        pretrained_path="resnet18.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet18"),
        model_args={"num_classes": 10},
    ).to(device)
    epochs = 20

    for use_kd in [True, False]:
        label = "KD" if use_kd else "CE"
        student = StudentConv().to(device)
        optimizer = optim.Adam(student.parameters(), lr=1e-3)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_student(
                student, teacher, train_loader, optimizer, device, use_kd
            )
            test_acc, test_loss = evaluate_metrics(student, test_loader, device)
            print(
                f"[{label}] Epoch {epoch:02d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%, test_loss={test_loss:.4f}"
            )
            writer.add_scalar(f"{label}/train_loss", train_loss, epoch)
            writer.add_scalar(f"{label}/train_acc", train_acc, epoch)
            writer.add_scalar(f"{label}/test_acc", test_acc, epoch)
            writer.add_scalar(f"{label}/test_loss", test_loss, epoch)

    writer.close()
    print(f"Done. Logs available in {args.logdir}.")


if __name__ == "__main__":
    main()
