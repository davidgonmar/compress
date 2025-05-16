import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, DataCollatorWithPadding
from compress.factorization.factorize import (
    to_low_rank_manual,
    all_same_svals_energy_ratio,
    all_same_rank_ratio,
    all_same_params_ratio,
    plot_singular_values,
)
from compress.flops import count_model_flops
import evaluate
import argparse
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess(batch, tokenizer, max_length):
    # Handles single-sentence and sentence-pair tasks
    if 'sentence2' in batch:
        tokens = tokenizer(
            batch['sentence1'],
            batch['sentence2'],
            max_length=max_length,
            truncation=True,
        )
    else:
        tokens = tokenizer(
            batch['sentence'],
            max_length=max_length,
            truncation=True,
        )
    tokens['labels'] = batch['label']
    return tokens


def evaluate_model(model, loader, metric, is_regression, device):
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        preds = outputs.logits.squeeze() if is_regression else torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(
            predictions=preds.cpu().numpy(),
            references=batch['labels'].cpu().numpy(),
        )
    return metric.compute()


def main():
    parser = argparse.ArgumentParser(description="Low-Rank BERT GLUE Evaluation")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--task_name", type=str, required=True,
                        choices=["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"])
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep_edge_layer", action="store_true",
                        help="Keep the classification head intact and not factorize it")
    parser.add_argument("--metric", type=str, default="energy",
                        choices=["energy","rank","params"],
                        help="Metric for selecting singular values: 'energy', 'rank', or 'params'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load GLUE dataset and metric
    raw = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    metric = evaluate.load("glue", args.task_name)

    # Tokenizer and data collator
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Prepare evaluation dataset
    split = 'validation_matched' if args.task_name == 'mnli' else 'validation'
    tokenized = raw[split].map(
        lambda ex: preprocess(ex, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=raw[split].column_names
    )
    eval_loader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    # Build and load model
    config = BertConfig.from_pretrained(
        args.model_name,
        num_labels=(1 if is_regression else raw['train'].features['label'].num_classes),
        finetuning_task=args.task_name,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )
    model.to(device)

    # Baseline evaluation
    baseline = evaluate_model(model, eval_loader, metric, is_regression, device)
    flops_base = count_model_flops(model, (3, args.max_seq_length), torch.long)
    n_params_base = sum(p.numel() for p in model.parameters())
    print(f"Baseline → FLOPs: {flops_base}, Params: {n_params_base}, Metrics: {baseline}")

    # Plot singular values of all convs/linears
    plot_singular_values(model)

    # Define grid
    energies = [0.9,0.99,0.999,0.9991,0.9993,0.9995,0.9997,0.9999,0.99993,0.99995,0.99997,0.99998,0.99999]
    ranks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    perms = ranks

    for x in (energies if args.metric=='energy' else ranks if args.metric=='rank' else perms):
        # Compute configuration
        if args.metric == 'energy':
            cfg = all_same_svals_energy_ratio(model, energy=x)
            tag = f"energy_{x:.5f}"
        elif args.metric == 'rank':
            cfg = all_same_rank_ratio(model, rank=x)
            tag = f"rank_{x:.2f}"
        else:
            cfg = all_same_params_ratio(model, ratio=x)
            tag = f"params_{x:.2f}"

        # Optionally keep classification head intact
        if args.keep_edge_layer and 'classifier' in cfg:
            del cfg['classifier']

        # Apply low-rank
        model_lr = to_low_rank_manual(model, cfg_dict=cfg, inplace=False)
        model_lr.to(device)

        # Evaluate
        res = evaluate_model(model_lr, eval_loader, metric, is_regression, device)
        flops_lr = count_model_flops(model_lr, (3, args.max_seq_length), torch.long)
        n_params_lr = sum(p.numel() for p in model_lr.parameters())

        print(f"{tag} → FLOPs: {flops_lr}, Params ratio: {n_params_lr/n_params_base:.4f}, Metrics: {res}")

        # Save factorized model
        torch.save(model_lr.state_dict(), f"{args.model_name}_{args.task_name}_{tag}.pth")

if __name__ == '__main__':
    main()
