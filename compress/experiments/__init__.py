import torch
from typing import Dict
import torch.nn as nn
import torchvision  # noqa
import compress.experiments.cifar_resnet # noqa


def evaluate_vision_model(
    model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, eval=True
) -> Dict[str, float]:
    prev_state = model.training
    if eval:
        model.eval()
    else:
        model.train()

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
    model.train(prev_state)
    return {"accuracy": accuracy, "loss": loss}


def mobilenetv2_cifar10_modifier(model):
    model.features[0][0].stride = (1, 1)
    for i in range(2, 7):
        model.features[i].conv[1][0].stride = (1, 1)
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 10),
    )
    return model


def resnet18_cifar10_modifier(model):
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.maxpool = nn.Identity()
    return model


def get_cifar10_modifier(model_str):
    if model_str == "mobilenet_v2":
        return mobilenetv2_cifar10_modifier
    elif model_str == "resnet18":
        return resnet18_cifar10_modifier
    else:
        raise ValueError(
            f"Model {model_str} not supported. Supported models: ['mobilenet_v2', 'resnet18']"
        )


def _maybe_identity(model, fn):
    if fn is None:
        return model
    else:
        return fn(model)


def load_vision_model(
    model_str: str,
    pretrained_path: str | None = None,
    strict=True,
    modifier_before_load=None,
    modifier_after_load=None,
    model_args={},
) -> torch.nn.Module:
    _d = {
        "resnet18": "torchvision.models.resnet18",
        "mobilenet_v2": "torchvision.models.mobilenet_v2",\
        # Depth must be one of 20, 32, 44, 56, 110, 1202"
        "resnet20": "compress.experiments.cifar_resnet.resnet20",
        "resnet32": "compress.experiments.cifar_resnet.resnet32",
        "resnet44": "compress.experiments.cifar_resnet.resnet44",
        "resnet56": "compress.experiments.cifar_resnet.resnet56",
        "resnet110": "compress.experiments.cifar_resnet.resnet110",
        "resnet1202": "compress.experiments.cifar_resnet.resnet1202",
    }

    if model_str not in _d:
        raise ValueError(
            f"Model {model_str} not supported. Supported models: {_d.keys()}"
        )

    model = eval(_d[model_str])(**model_args)

    if pretrained_path is None and modifier_after_load is not None:
        raise ValueError(
            "modifier_after_load should be None if pretrained_path is None"
        )

    model = _maybe_identity(model, modifier_before_load)
    if pretrained_path:
        loaded = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model" in loaded:
            model.load_state_dict(loaded["model"], strict=strict)
        if isinstance(loaded, dict) and "_orig_mod" in next(
            iter(loaded.keys())
        ):  # torch.compile model:
            new_dict = {}
            for k, v in loaded.items():
                if k.startswith("_orig_mod."):
                    new_dict[k[len("_orig_mod.") :]] = v
                else:
                    new_dict[k] = v
            model.load_state_dict(new_dict, strict=strict)
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded, strict=strict)
        elif isinstance(loaded, nn.Module):
            model.load_state_dict(loaded.state_dict(), strict=strict)
        else:
            raise ValueError("Loaded model is not a dict or nn.Module")
    model = _maybe_identity(model, modifier_after_load)
    return model


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2470, 0.2435, 0.2616]

# This is incorrect, because it was computed by averaging the per-batch standard deviations.
incorrect_cifar10_std = [0.2023, 0.1994, 0.2010]
