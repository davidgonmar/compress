import argparse
import logging
import os
import random

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset
import evaluate
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)

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
        description="Fine-tune BERT on GLUE tasks using Hugging Face Evaluate"
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
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

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

    model_dir = os.path.join(args.output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, "train.json")

    history = []

    for epoch in range(1, args.num_train_epochs + 1):

        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                avg = total_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch} Step {step+1}/{len(train_dataloader)} Loss {avg:.4f}"
                )

        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} finished. Avg loss {avg_train_loss:.4f}")

        model.eval()
        metric = evaluate.load("glue", args.task_name)
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            preds = logits.squeeze() if is_regression else torch.argmax(logits, dim=-1)
            metric.add_batch(
                predictions=preds.cpu().numpy(),
                references=batch["labels"].cpu().numpy(),
            )
        results = metric.compute()
        logger.info(f"Validation results for epoch {epoch}: {results}")

        history.append({"epoch": epoch, "train_loss": avg_train_loss, **results})

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logger.info(f"Saved model and tokenizer to {model_dir}")

    with open(results_file, "w") as f:
        json.dump({"history": history}, f, indent=2)
    logger.info(f"Saved training history to {results_file}")


if __name__ == "__main__":
    main()
