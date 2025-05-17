import argparse
import json
import random

import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

from compress.quantization import to_quantized_offline, get_activations_transformers
from compress.quantization.recipes import get_generic_recipe_quant

args = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline quantize BERT on GLUE and evaluate across bit-widths"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="GLUE task name (e.g., cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Pretrained BERT model name or path",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Optional path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--leave_edge_layers_8_bits",
        action="store_true",
        help="Force first/last layers to 8-bit",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation and calibration",
    )
    parser.add_argument(
        "--calibration_batches",
        type=int,
        default=4,
        help="Number of batches for calibration",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="quantization_results.json",
        help="Output JSON file for results",
    )
    return parser.parse_args()


def collate_fn(batch, tokenizer):
    return DataCollatorWithPadding(tokenizer)(batch)


def prepare_dataloaders(task_name, tokenizer, batch_size, calibration_batches):
    raw = load_dataset("glue", task_name)
    key_map = {
        "cola": ("sentence", None),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    s1, s2 = key_map[task_name]

    def preprocess(example):
        args = {
            "padding": True,
            "truncation": True,
            "max_length": 128,
        }
        if s2 is None:
            encoded = tokenizer(example[s1], **args)
        else:
            encoded = tokenizer(example[s1], example[s2], **args)
        encoded["labels"] = example["label"]
        return encoded

    tokenized = raw.map(
        preprocess, batched=True, remove_columns=raw["train"].column_names
    )

    # calibration subset
    train_dataset = tokenized["train"]
    # select calibration_batches * batch_size random samples
    total = min(len(train_dataset), calibration_batches * batch_size)
    indices = random.sample(range(len(train_dataset)), total)
    calib_subset = Subset(train_dataset, indices)
    calib_loader = DataLoader(
        calib_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    # eval dataset
    eval_split = "validation_matched" if task_name == "mnli" else "validation"
    eval_dataset = tokenized[eval_split]
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )
    return calib_loader, eval_loader


def evaluate_model(model, dataloader, device, is_regression=False):
    metric = evaluate.load("glue", args.task_name)
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        preds = logits.squeeze() if is_regression else torch.argmax(logits, dim=-1)
        metric.add_batch(
            predictions=preds.cpu().numpy(),
            references=batch["labels"].cpu().numpy(),
        )
    return metric.compute()


def main():
    global args
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Load model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name if args.pretrained_path is None else args.pretrained_path,
        num_labels=(
            1
            if args.task_name == "stsb"
            else load_dataset("glue", args.task_name)["train"]
            .features["label"]
            .num_classes
        ),
        finetuning_task=args.task_name,
    )
    model.to(device)

    calib_loader, eval_loader = prepare_dataloaders(
        args.task_name, tokenizer, args.batch_size, args.calibration_batches
    )
    is_regression = args.task_name == "stsb"

    # Evaluate original
    orig_results = evaluate_model(model, eval_loader, device, is_regression)
    results = [{"type": "original", **orig_results}]

    # calibration activations
    activations = get_activations_transformers(
        model,
        calib_loader,
        specs=get_generic_recipe_quant(
            model,
            bits_activation=8,
            bits_weight=8,
            clip_percentile=0.995,
            leave_edge_layers_8_bits=args.leave_edge_layers_8_bits,
            symmetric=False,
        ),
    )

    bit_widths = [2, 4, 8, 16]
    for w_bits in bit_widths:
        for act_bits in bit_widths:
            spec = get_generic_recipe_quant(
                model,
                bits_activation=act_bits,
                bits_weight=w_bits,
                clip_percentile=0.995,
                leave_edge_layers_8_bits=args.leave_edge_layers_8_bits,
                symmetric=False,
            )
            quanted = to_quantized_offline(
                model, spec, activations=activations, inplace=False
            )
            print(quanted)
            quanted.to(device)
            res = evaluate_model(quanted, eval_loader, device, is_regression)
            print(f"W{w_bits}A{act_bits}: {res}")
            results.append(
                {
                    "type": f"W{w_bits}A{act_bits}",
                    "leave_edge_layers_8_bits": args.leave_edge_layers_8_bits,
                    **res,
                }
            )

    # Write results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Quantization results saved to {args.output_file}")


if __name__ == "__main__":
    main()
