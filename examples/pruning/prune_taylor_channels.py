import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
    cifar10_mean,
    cifar10_std,
)
from compress.sparsity.prune import (
    prune_channels_resnet20_policies,
    get_sparsity_information_str,
    TaylorExpansionInterPruner,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ]
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dataloader = DataLoader(testset, batch_size=32, shuffle=False)
    model = load_vision_model(
        "resnet20",
        pretrained_path="resnet20.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet20"),
        model_args={"num_classes": 10},
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy before pruning: {ret['accuracy']:.2f}%")
    print(f"Loss before pruning: {ret['loss']:.4f}")
    sparsity = 0.4
    policies = prune_channels_resnet20_policies(
        {
            "name": "sparsity_ratio",
            "value": sparsity,
        }
    )

    data_iter = iter(dataloader)

    def runner():
        nonlocal data_iter
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        return torch.nn.functional.cross_entropy(outputs, targets)

    pruner = TaylorExpansionInterPruner(model, policies, runner, n_iters=30)
    model = pruner.prune()
    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy after pruning: {ret['accuracy']:.2f}%")
    print(f"Loss after pruning: {ret['loss']:.4f}")
    print(f"Sparsity information: {get_sparsity_information_str(model)}")


if __name__ == "__main__":
    main()
