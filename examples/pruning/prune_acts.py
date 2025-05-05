import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    evaluate_vision_model,
)
from compress.sparsity.prune import (
    unstructured_resnet18_policies,
    measure_nonzero_params,
    WandaPruner,
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
        pretrained_path="resnet18.pth",
        strict=True,
        modifier_before_load=get_cifar10_modifier("resnet18"),
        model_args={"num_classes": 10},
    ).to(device)

    nparams = sum(p.numel() for p in model.parameters())
    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy before pruning: {ret['accuracy']:.2f}%")
    print(f"Loss before pruning: {ret['loss']:.4f}")

    sparsity = 0.3
    policies = unstructured_resnet18_policies(sparsity)

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
    pruner = WandaPruner(model, policies, runner, n_iters=60)
    model = pruner.prune()

    ret = evaluate_vision_model(model, dataloader)
    print(f"Accuracy after pruning: {ret['accuracy']:.2f}%")
    print(f"Loss after pruning: {ret['loss']:.4f}")
    print(
        f"Percent of non-zero parameters after pruning: {measure_nonzero_params(model) / nparams:.2%}"
    )


if __name__ == "__main__":
    main()
