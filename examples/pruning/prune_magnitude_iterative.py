import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
)
from compress.sparsity.prune import (
    unstructured_resnet18_policies,
    MagnitudePruner,
    measure_nonzero_params,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)

    model = load_vision_model(
        "resnet18",
        pretrained_path="resnet18.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet18"),
        model_args={"num_classes": 10},
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
    )

    target_sparsity = 0.02
    n_iters = 7
    epochs_per_iter = 1

    for it in range(1, n_iters + 1):
        print(f"\n=== Iteration {it}/{n_iters} ===")

        policies = unstructured_resnet18_policies(
            {
                "name": "sparsity_ratio",
                "value": target_sparsity
                + (1 - target_sparsity) * ((n_iters - it) / n_iters),
            },
            normalize_non_prunable=True,
        )
        # print(policies)
        pruner = MagnitudePruner(model, policies)
        model = pruner.prune()

        nonzero = measure_nonzero_params(model)
        print(
            f"  After pruning — nonzero params: {nonzero}/{total_params} "
            f"({nonzero/total_params:.2%} remain)"
        )

        for epoch in range(1, epochs_per_iter + 1):
            loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            print(f"    Epoch {epoch}/{epochs_per_iter} — train loss: {loss:.4f}")
            stats = evaluate_vision_model(model, testloader)
            print(
                f"  Val accuracy: {stats['accuracy']:.2f}% | loss: {stats['loss']:.4f}"
            )

    final_nonzero = measure_nonzero_params(model)
    print("\n=== Final Results ===")
    print(f"Accuracy : {stats['accuracy']:.2f}%")
    print(f"Loss     : {stats['loss']:.4f}")
    print(f"Sparsity : {(1 - final_nonzero/total_params):.2%} pruned")


if __name__ == "__main__":
    main()
