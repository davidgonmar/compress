import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
)
from compress.sparsity.prune import (
    prune_channels_resnet18_policies,
    GroupedActivationPruner,
    measure_nonzero_params,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dataloader = DataLoader(testset, batch_size=32, shuffle=False)

    model = load_vision_model(
        "resnet18",
        pretrained_path="sparsity_model.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet18"),
        model_args={"num_classes": 10},
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy before pruning: {ret['accuracy']:.2f}%")
    print(f"Loss before pruning: {ret['loss']:.4f}")

    policies = prune_channels_resnet18_policies(
        {
            "name": "threshold",
            "value": 25.0,
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

    # Use WandaPruner here
    pruner = GroupedActivationPruner(model, policies, runner, n_iters=100)
    model = pruner.prune()

    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy after pruning: {ret['accuracy']:.2f}%")
    print(f"Loss after pruning: {ret['loss']:.4f}")
    print(
        f"Percent of non-zero parameters after pruning: {measure_nonzero_params(model) / nparams:.2%}"
    )


if __name__ == "__main__":
    main()
