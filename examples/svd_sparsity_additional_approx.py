import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import argparse
from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torchvision
from tqdm import tqdm
from compress.factorize import hoyer_svd_sparsity_grad_adder_given_svds
from compress.regularizers import singular_values_hoyer_sparsity
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default="cifar_resnet18_model.pth")
parser.add_argument("--sv_regularizer", type=str, default="noop")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--regularizer_weight", type=float, default=1.0)
parser.add_argument("--regularizer_scheduler", type=str, default="noop")
parser.add_argument("--finetune", action="store_true")
args = parser.parse_args()


def get_cifar10():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = datasets.CIFAR10(
        root="data", train=True, transform=transform_train, download=True
    )
    test_dataset = datasets.CIFAR10(
        root="data", train=False, transform=transform_test, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    return train_loader, test_loader, {"input_size": (3, 32, 32), "num_classes": 10}


train_loader, val_loader, model_params = get_cifar10()

model = resnet18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
sched = StepLR(optimizer, step_size=10, gamma=0.2)

if args.finetune:
    torch_weights = torchvision.models.resnet18(pretrained=True).state_dict()
    del torch_weights["fc.weight"]
    del torch_weights["fc.bias"]
    model.load_state_dict(torch_weights, strict=False)

weights = args.regularizer_weight


def weight_schedule_noop(epochnum):
    return weights


def weight_schedule_exp(epochnum):
    return 0.5 * weights * 2 ** (epochnum / args.epochs)


weight_sched = {
    "noop": weight_schedule_noop,
    "exp": weight_schedule_exp,
}[args.regularizer_scheduler]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


svd = {}

for name, param in model.named_parameters():
    if param.dim() == 4 or "weight" in name:
        if param.dim() == 4:
            param_rs = param.reshape(param.shape[0], -1)
        elif param.dim() == 2:
            param_rs = param
        else:
            continue
        U, S, Vt = torch.linalg.svd(param_rs, full_matrices=False)
        svd[name] = (U, S, Vt)


old_params = None
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    idxx = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        total_loss = loss
        total_loss.backward()

        old_params = copy.deepcopy(model.state_dict())
        hoyer_svd_sparsity_grad_adder_given_svds(
            model.named_parameters(), svd, weight_sched(epoch)
        )

        optimizer.step()

        if idxx % 5 == 0:
            # recompute svd
            for name, param in model.named_parameters():
                if param.dim() == 4 or "weight" in name:
                    if param.dim() == 4:
                        param_rs = param.reshape(param.shape[0], -1)
                    elif param.dim() == 2:
                        param_rs = param
                    else:
                        continue
                    U, S, Vt = torch.linalg.svd(param_rs, full_matrices=False)
                    svd[name] = (U, S, Vt)
        else:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in svd and param.grad is not None:
                        if param.dim() == 4:
                            grad_rs = old_params[name].reshape(
                                param.shape[0], -1
                            ) - param.reshape(param.shape[0], -1)
                        elif param.dim() == 2:
                            grad_rs = old_params[name] - param
                        else:
                            continue
                        U, S, Vt = svd[name]
                        # clamp

                        deltaW_V = torch.matmul(grad_rs, Vt.T)
                        U_delta = torch.matmul(U.T, deltaW_V)

                        eps = 1e-3
                        s_diff = S.view(-1, 1) - S.view(1, -1)
                        print(s_diff)

                        mask = ~torch.eye(len(S), dtype=bool, device=S.device)
                        UV_delta = U_delta / s_diff
                        UV_delta[~mask] = 0.0
                        delta_U = torch.matmul(U, UV_delta)
                        U_approx = U - delta_U

                        deltaW_T_U = torch.matmul(grad_rs.T, U)
                        V_delta = torch.matmul(Vt, deltaW_T_U)
                        V_delta = V_delta / s_diff
                        V_delta[~mask] = 0.0
                        delta_V = torch.matmul(Vt.T, V_delta)
                        V_approx = Vt.T - delta_V

                        # s_approx = S - torch.einsum("i,ij,j->i", U, grad_rs, Vt)
                        s_approx = S - torch.einsum(
                            "ij,ij->j", U, torch.matmul(grad_rs, Vt.transpose(0, 1))
                        )
                        print("sapprox", s_approx)
                        svd[name] = (U_approx, s_approx, V_approx.T)

        train_loss += loss.item() * x.size(0)
        idxx += 1
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")

    # measure hoyer sparsity of svd vals
    li = []
    for param in model.parameters():
        if param.dim() not in [2, 4]:
            continue
        param_frob = param.reshape(-1).pow(2).sum().sqrt()
        if param.dim() == 4:
            param_rs = param.reshape(param.shape[0], -1)
        else:
            param_rs = param
        r = singular_values_hoyer_sparsity(param_rs, True)
        li.append(r.item())

    print(
        f"Epoch {epoch+1}/{args.epochs}, Hoyer Sparsity: {[f'{val:.6f}' for val in li]}"
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
        torch.save(model, args.save_path)

print("Finished training. Saving model...")
torch.save(model, args.save_path)
