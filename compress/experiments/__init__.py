import torch
from typing import Dict


def evaluate_vision_model(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader
) -> Dict[str, float]:
    device = next(model.parameters()).device

    correct = 0
    total = 0
    loss = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss += criterion(outputs, labels).item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss = loss / total

    return {"accuracy": accuracy, "loss": loss}
