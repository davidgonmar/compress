import argparse
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from compress.experiments import (
    load_vision_model,
    get_cifar10_modifier,
    cifar10_mean,
    cifar10_std,
    evaluate_vision_model,
)
from compress.layer_fusion import fuse_conv_bn, resnet20_fuse_pairs
from compress.sparsity.policy import Metric
from compress.sparsity.prune import (
    ActivationNormInterGroupPruner,
    TaylorExpansionInterGroupPruner,
    WeightNormInterGroupPruner,
    fuse_bn_conv_sparse_train,
    get_sparsity_information,
)
from compress.sparsity.recipes import per_output_channel_resnet20_policy_dict
from compress.sparsity.runner import VisionClassificationModelRunner
from compress import seed_everything


def main():
    parser = argparse.ArgumentParser("One‑shot pruning")
    parser.add_argument("--model", default="resnet20")
    parser.add_argument("--pretrained_path", default="resnet20.pth")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_sparsity", type=float, default=0.5)
    parser.add_argument(
        "--method",
        choices=["norm_weights", "norm_activations", "taylor_no_bias", "taylor_bias"],
        default="norm_weights",
    )
    parser.add_argument("--calibration_samples", type=int, default=512)
    parser.add_argument("--calibration_bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats_file", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cifar10_mean, cifar10_std)]
    )
    testset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = load_vision_model(
        args.model,
        pretrained_path=args.pretrained_path,
        strict=True,
        modifier_before_load=get_cifar10_modifier(args.model),
    )
    model = fuse_conv_bn(
        model, resnet20_fuse_pairs, fuse_impl=fuse_bn_conv_sparse_train
    ).to(device)

    baseline = evaluate_vision_model(model, testloader)

    policies = per_output_channel_resnet20_policy_dict(
        inter_metric=Metric(name="sparsity_ratio", value=args.target_sparsity),
    )

    if args.method == "norm_weights":
        pruner = WeightNormInterGroupPruner(model, policies)
    elif args.method == "norm_activations":
        calib_subset = torch.utils.data.Subset(
            datasets.CIFAR10("./data", train=True, download=True, transform=transform),
            torch.randint(0, len(testset), (args.calibration_samples,)),
        )
        calib_loader = DataLoader(calib_subset, batch_size=args.calibration_bs)
        runner = VisionClassificationModelRunner(model, calib_loader)
        pruner = ActivationNormInterGroupPruner(model, policies, runner)
    else:
        calib_subset = torch.utils.data.Subset(
            datasets.CIFAR10("./data", train=True, download=True, transform=transform),
            torch.randint(0, len(testset), (args.calibration_samples,)),
        )
        # needs bs=1 for taylor estimation
        calib_loader = DataLoader(calib_subset, batch_size=1)
        runner = VisionClassificationModelRunner(model, calib_loader)
        pruner = TaylorExpansionInterGroupPruner(
            model,
            policies,
            runner,
            approx="fisher_diag",
            use_bias=args.method == "taylor_bias",
        )

    model = pruner.prune()

    sparsity_info = get_sparsity_information(model)
    pruned = evaluate_vision_model(model, testloader)

    stats = {
        "config": vars(args),
        "baseline": baseline,
        "pruned": {
            "accuracy": pruned["accuracy"],
            "loss": pruned["loss"],
            "sparsity": sparsity_info,
        },
    }
    with open(args.stats_file, "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
