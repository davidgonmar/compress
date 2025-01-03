import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torch.optim import Adam
from compress.regularizers import SingularValuesRegularizer
from examples.utils.models import SimpleMNISTModel
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="mnist_model.pth")
parser.add_argument("--sv_regularizer", type=str, default="noop")
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--regularizer_weight", type=float, default=1.0)
args = parser.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

model = SimpleMNISTModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

kwargs = {
    "entropy": {},
    "hoyer_sparsity": {"normalize": True},
    "scad": {"lambda_val": 0.1, "a_val": 3.7},
}  # SCAD needs tuning
regularizer = SingularValuesRegularizer(
    metric=args.sv_regularizer,
    params=[model.model[1].weight, model.model[3].weight],
    weights=args.regularizer_weight,
    **kwargs[args.sv_regularizer],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0
    for batch in train_loader:
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
        torch.save(model.state_dict(), args.save_path)


print("Finished training. Saving model...")
torch.save(model.state_dict(), args.save_path)
