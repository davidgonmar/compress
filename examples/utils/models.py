import torch


class SimpleMNISTModel(torch.nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.model(x)
