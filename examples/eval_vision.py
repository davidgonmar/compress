from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean,
    cifar10_std,
)

import torch
from torchvision import transforms, datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ]
)

test_dataset = datasets.CIFAR10(
    root="data", train=False, transform=transform, download=True
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = load_vision_model(
    "resnet20",
    pretrained_path="resnet20.pth",
    strict=True,
    modifier_before_load=get_cifar10_modifier("resnet20"),
    modifier_after_load=None,
    model_args={"num_classes": 10},
).to(device)

print(model)

result = evaluate_vision_model(model, test_loader)

print(f"Accuracy: {result['accuracy']:.2f}%")
print(f"Loss: {result['loss']:.4f}")
