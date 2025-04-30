import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from compress.quantization import (
    AutoBitAllocationLinear,
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP()
model.load_state_dict(torch.load("mlp.pth", map_location=device))
model = model.to(device)


def replace_linear_with_quant(module, w_spec, a_spec):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, AutoBitAllocationLinear(w_spec, a_spec, child))
            # freeze the weights and biases
            child.weight.requires_grad = False
            child.bias.requires_grad = False
        else:
            replace_linear_with_quant(child, w_spec, a_spec)


w_spec = IntAffineQuantizationSpec(
    nbits=8,
    signed=True,
    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
    percentile=0.995,
)
a_spec = IntAffineQuantizationSpec(
    nbits=8,
    signed=True,
    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
    percentile=0.995,
)
replace_linear_with_quant(model, w_spec, a_spec)

model.to(device)
transform = transforms.ToTensor()
trainset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True)
testset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
bit_penalty_lambda = 2e-3


def train(model, dataloader):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        bit_penalty = sum(
            (m.b_w + m.b_a)
            for m in model.modules()
            if isinstance(m, AutoBitAllocationLinear)
        )

        loss += bit_penalty_lambda * bit_penalty

        loss.backward()
        # print b grads
        """for m in model.modules():
            if isinstance(m, AutoBitAllocationLinear):
                print("b_w grad:", m.b_w.grad)
                print("b_a grad:", m.b_a.grad)
        """
        optimizer.step()


for epoch in range(1000):
    train(model, trainloader)
    print(f"Fine-tune Epoch {epoch + 1} complete ✅")
    print("Printing bits:")
    for m in model.modules():
        if isinstance(m, AutoBitAllocationLinear):
            print("w:", m.b_w.item(), "a:", m.b_a.item())

    # evaluate the model
    loss = 0
    correct = 0
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    print(f"Test Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
