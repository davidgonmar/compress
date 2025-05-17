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
    WandaPruner,
    get_sparsity_information_str,
    make_vision_runner,
)
from compress.sparsity.schedulers import get_scheduler

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
    import argparse

    parser = argparse.ArgumentParser(description="Iterative Magnitude Pruning")
    parser.add_argument("--model", type=str, default="resnet20")
    parser.add_argument("--pretrained_path", type=str, default="resnet20.pth")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--test_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--target_sparsity", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=12)
    parser.add_argument("--epochs_per_iter", type=int, default=5)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument(
        "--method",
        type=str,
        choices=["magnitude", "wanda"],
        default="wanda",
        help="Pruning method to use: 'magnitude' or 'wanda'"
    )
    parser.add_argument("--wanda_samples", type=int, default=512)
    parser.add_argument("--wanda_n_iters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    scheduler = get_scheduler(args.scheduler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )

    trainset = datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform_test
    )
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.train_workers
    )
    testloader = DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.test_workers
    )

    model = load_vision_model(
        args.model,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model),
        model_args={"num_classes": args.num_classes},
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    for it in range(1, args.n_iters + 1):
        print(f"\n=== Iteration {it}/{args.n_iters} ===")

        policies = unstructured_resnet18_policies(
            {
                "name": "sparsity_ratio",
                "value": scheduler(it, args.n_iters, args.target_sparsity),
            },
            normalize_non_prunable=True,
        )
        if args.method == "magnitude":
            pruner = MagnitudePruner(model, policies)
        elif args.method == "wanda":
            pruner = WandaPruner(
                model,
                policies,
                make_vision_runner(
                    model,
                    DataLoader(
                        torch.utils.data.Subset(
                            trainset,
                            torch.randint(0, len(trainset), (args.wanda_samples,))
                        ),
                        batch_size=args.batch_size,
                        shuffle=True,
                    ),
                    criterion,
                    device,
                ),
                n_iters=args.wanda_n_iters,
            )
        else:
            raise ValueError(f"Unknown pruning method: {args.method}")

        model = pruner.prune()
        pruner = pruner.dispose()
        print("  Pruning done.")
        stats = evaluate_vision_model(model, testloader)
        print(
            f"  Val accuracy before ft: {stats['accuracy']:.2f}% | loss: {stats['loss']:.4f}"
        )

        print("sparsity_info:", get_sparsity_information_str(model))

        for epoch in range(1, args.epochs_per_iter + 1):
            loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            print(f"    Epoch {epoch}/{args.epochs_per_iter} — train loss: {loss:.4f}")
            stats = evaluate_vision_model(model, testloader)
            print(
                f"  Val accuracy: {stats['accuracy']:.2f}% | loss: {stats['loss']:.4f}"
            )

    print(get_sparsity_information_str(model))

if __name__ == "__main__":
    main()
