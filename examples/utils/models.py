import torch
from typing import Tuple
from functools import reduce, partial


product = partial(reduce, lambda x, y: x * y)


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_size: Tuple[int, ...], num_classes: int):
        super(MLPClassifier, self).__init__()
        self.num_classes, self.input_size = num_classes, input_size
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(product(input_size), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ConvClassifier(torch.nn.Module):
    def __init__(self, input_size: Tuple[int, ...], num_classes: int):
        super(ConvClassifier, self).__init__()
        self.num_classes, self.input_size = num_classes, input_size
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[0], 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.LazyLinear(num_classes),
        )

    def forward(self, x):
        return self.model(x)
