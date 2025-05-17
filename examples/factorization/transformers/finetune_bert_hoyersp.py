import argparse
import logging
import os
import random
import math
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from datasets import load_dataset
import evaluate
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)

# Hoyer regularizer imports
from compress.factorization.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
)
from compress.factorization.utils import matrix_approx_rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEY_MAPPING = {
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT on GLUE tasks with Hoyer singular-value regularization"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=list(KEY_MAPPING.keys()),
        help="The GLUE task to train on",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=128, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for train and eval"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Initial learning rate"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for initialization"
    )
    parser.add_argument(
        "--start_reg",
        type=float,
        default=0.5,
        help="Initial regularization weight (cosine schedule start)",
    )
    parser.add_argument(
        "--end_reg",
        type=float,
        default=0.0,
        help="Final regularization weight (cosine schedule end)",
    )
    parser.add_argument(
        "--T0",
        type=int,
        default=3,
        help="Number of epochs for the first regularization annealing cycle",
    )
    parser.add_argument(
        "--T_mult",
        type=int,
        default=1,
        help="Multiplicative factor for subsequent cycles in regularization annealing",
    )
    return parser.parse_args()


def weight_schedule(epoch, start, end, T_0, T_mult):
    T_i = T_0
    ep_i = epoch
    while ep_i >= T_i:
        ep_i -= T_i
        T_i *= T_mult
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * ep_i / T_i))


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load dataset and tokenizer/model
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    num_labels = (
        1 if is_regression else raw_datasets["train"].features["label"].num_classes
    )

    config = BertConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        finetuning_task=args.task_name,
    )
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare regularizer on linear layers
    params_and_reshapers = extract_weights_and_reshapers(
        model,
        cls_list=(torch.nn.Linear,),
        keywords={"weight"},
    )
    regularizer = SingularValuesRegularizer(
        metric="hoyer_sparsity",
        params_and_reshapers=params_and_reshapers,
        weights=1.0,
        normalize=False,
    )

    # Preprocessing
    sentence1_key, sentence2_key = KEY_MAPPING[args.task_name]

    def preprocess_function(examples):
        if sentence2_key is None:
            tokens = tokenizer(
                examples[sentence1_key],
                padding="max_length",
                max_length=args.max_seq_length,
                truncation=True,
            )
        else:
            tokens = tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                padding="max_length",
                max_length=args.max_seq_length,
                truncation=True,
            )
        if "label" in examples:
            tokens["labels"] = examples["label"]
        return tokens

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    data_collator = DataCollatorWithPadding(tokenizer)
    train_dataset = tokenized_datasets["train"]
    eval_split = "validation_matched" if args.task_name == "mnli" else "validation"
    eval_dataset = tokenized_datasets[eval_split]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    # Optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training
    history = []
    metric = evaluate.load("glue", args.task_name)

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0
        total_reg = 0.0
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for step, batch in enumerate(train_pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # compute regularization
            reg_w = weight_schedule(
                epoch - 1, args.start_reg, args.end_reg, args.T0, args.T_mult
            )
            reg = regularizer()
            loss = loss + reg_w * reg
            loss.backward()
            total_loss += outputs.loss.item()
            total_reg += reg.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = total_loss / step
            avg_reg = total_reg / step
            train_pbar.set_postfix(loss=f"{avg_loss:.4f}", reg=f"{avg_reg:.4f}")

        logger.info(
            f"Epoch {epoch} finished. Avg loss {avg_loss:.4f} Avg reg {avg_reg:.4f}"
        )

        # Evaluation
        model.eval()
        eval_pbar = tqdm(eval_dataloader, desc=f"Epoch {epoch} Eval", leave=False)
        for batch in eval_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            preds = logits.squeeze() if is_regression else torch.argmax(logits, dim=-1)
            metric.add_batch(
                predictions=preds.cpu().numpy(),
                references=batch["labels"].cpu().numpy(),
            )
        print("Approximate ranks per layer:")

        def _mul(shape):
            res = 1
            for d in shape:
                res *= d
            return res

        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                approx = matrix_approx_rank(module.weight)
                total = _mul(module.weight.shape)
                print(
                    f"{name}: rank {approx} / {module.weight.shape[0]}, total elems {total}"
                )
        results = metric.compute()
        logger.info(f"Validation results for epoch {epoch}: {results}")
        history.append(
            {"epoch": epoch, "train_loss": avg_loss, "train_reg": avg_reg, **results}
        )

    # Save model, tokenizer, and history
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(args.output_dir, "model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))
    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump({"history": history}, f, indent=2)

    logger.info(f"Saved model, tokenizer, and history to {args.output_dir}")


if __name__ == "__main__":
    main()
