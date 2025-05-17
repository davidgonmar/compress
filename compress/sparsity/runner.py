import torch
import torch.nn as nn


class Runner:
    def __init__(self):
        pass

    def iteration(self) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses")


class VisionClassificationModelRunner(Runner):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.device = model.device
        self.dataloader_iter = iter(dataloader)
        self.criterion = nn.CrossEntropyLoss()

    def iteration(self):
        image, label = next(self.dataloader_iter)
        model_outs = self.model(image.to(self.device))
        loss = self.criterion(model_outs, label.to(self.device))
        return {
            "model_outs": model_outs,
            "loss": loss,
            "data": (image, label),
        }
