import argparse
import os
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)
from compress.factorization.factorize import to_low_rank_global
from compress.flops import count_model_flops
from compress.utils import get_all_convs_and_linears

# -- Helpers ------------------------------------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -- Argument Parsing --------------------------------------------------------

parser = argparse.ArgumentParser(
    description="BERT + Global Low-Rank Factorization on GLUE"
)
parser.add_argument(
    "--task_name",
    type=str,
    required=True,
    choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
)
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--output_dir", type=str, default="./factorized_outputs")
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75, 0.9]
)
parser.add_argument(
    "--keep_edge_layer",
    action="store_true",
    help="Whether to keep the first & last linear layers intact",
)
parser.add_argument(
    "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
args = parser.parse_args()

# -- Setup --------------------------------------------------------------------

set_seed(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# -- Load Data & Metric ------------------------------------------------------

raw_datasets = load_dataset("glue", args.task_name)
is_regression = args.task_name == "stsb"
metric = evaluate.load("glue", args.task_name)

tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
data_collator = DataCollatorWithPadding(tokenizer)


def preprocess(examples):
    # single vs. pair
    if (
        raw_datasets["train"].column_names.count("sentence1") > 0
        and raw_datasets["train"].column_names.count("sentence2") > 0
    ):
        toks = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            max_length=args.max_seq_length,
            truncation=True,
        )
    else:
        toks = tokenizer(
            examples["sentence"], max_length=args.max_seq_length, truncation=True
        )
    toks["labels"] = examples["label"]
    return toks


tokenized = raw_datasets.map(
    preprocess, batched=True, remove_columns=raw_datasets["train"].column_names
)
train_loader = DataLoader(
    tokenized["train"],
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=data_collator,
)
eval_split = "validation_matched" if args.task_name == "mnli" else "validation"
eval_loader = DataLoader(
    tokenized[eval_split],
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

# -- Load Model --------------------------------------------------------------

config = BertConfig.from_pretrained(
    args.model_name,
    num_labels=(
        1 if is_regression else raw_datasets["train"].features["label"].num_classes
    ),
    finetuning_task=args.task_name,
)
model = BertForSequenceClassification.from_pretrained(args.model_name, config=config)
model.to(args.device)

# -- Baseline Evaluation -----------------------------------------------------


def evaluate_model(m):
    m.eval()
    for batch in eval_loader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            out = m(**batch)
        preds = (
            out.logits.squeeze() if is_regression else torch.argmax(out.logits, axis=-1)
        )
        metric.add_batch(
            predictions=preds.cpu().numpy(), references=batch["labels"].cpu().numpy()
        )
    return metric.compute()


print(model)
baseline_stats = {
    "flops": count_model_flops(model, (3, args.max_seq_length), dtype=torch.int64),
    "metrics": evaluate_model(model),
}
print(
    f"Baseline → FLOPs: {baseline_stats['flops']}, Metrics: {baseline_stats['metrics']}"
)

# -- Gather all linear layers only ------------------------------------------

linear_keys = get_all_convs_and_linears(model)


if args.keep_edge_layer:
    linear_keys = [
        k for k in linear_keys if "encoder.layer.0" not in k and "classifier" not in k
    ]

# -- Apply Global Low-Rank and Re-evaluate ----------------------------------

results = {"baseline": baseline_stats, "factorized": {}}

for r in args.ratios:
    # factorize (non-inplace)
    m_lr = to_low_rank_global(
        model,
        sample_input=torch.zeros((3, args.max_seq_length)).long(),
        ratio_to_keep=r,
        inplace=False,
        keys=linear_keys,
        metric="flops",
        keep_edge_layer=args.keep_edge_layer,
    )
    m_lr.to(args.device)

    # eval
    fl = count_model_flops(m_lr, (3, args.max_seq_length), torch.long)
    met = evaluate_model(m_lr)
    param_ratio = sum(p.numel() for p in m_lr.parameters()) / sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"Ratio {r:.2f} → FLOPs: {fl}, Params ratio: {param_ratio:.4f}, Metrics: {met}"
    )
    results["factorized"][f"{r:.2f}"] = {
        "flops": fl,
        "param_ratio": param_ratio,
        "metrics": met,
    }

    # save
    torch.save(m_lr.state_dict(), os.path.join(args.output_dir, f"bert_lr_{r:.2f}.pth"))

# -- Dump Results ------------------------------------------------------------

with open(os.path.join(args.output_dir, "factorization_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nAll done! Results saved to {args.output_dir}/factorization_results.json")
