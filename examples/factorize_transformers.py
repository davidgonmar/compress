import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import copy
import argparse
from compress.factorize import to_low_rank_global, to_low_rank

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="textattack/bert-base-uncased-imdb"
)
parser.add_argument("--dataset_name", type=str, default="imdb")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--print_model", action="store_true")
parser.add_argument("--keep_last_layer", action="store_true")
parser.add_argument("--do_global", action="store_true")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

dataset = load_dataset(args.dataset_name)
if "test" not in dataset:
    raise ValueError(
        f"The dataset {args.dataset_name} does not contain a 'test' split."
    )

dataset = dataset.map(
    lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True
)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

dataset["test"] = dataset["test"].select(range(1000))
test_loader = DataLoader(dataset["test"], batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def evaluate(model, loader, device):
    model.eval()
    metric = load_metric("accuracy")
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=preds, references=labels)

    avg_loss = total_loss / len(loader)
    accuracy = metric.compute()["accuracy"]
    return avg_loss, accuracy


initial_loss, initial_accuracy = evaluate(model, test_loader, device)
nparams_orig = sum(p.numel() for p in model.parameters())
print(
    f"Initial Test Loss: {initial_loss:.4f}, Test Accuracy: {initial_accuracy:.4f}, Number of Parameters: {nparams_orig}"
)


def should_compress(module, name):
    return isinstance(module, torch.nn.Linear) and (
        not args.keep_last_layer or "classifier" not in name
    )


compression_fn = to_low_rank_global if args.do_global else to_low_rank
ratios = [0.05, 0.075, 0.09, 0.1, 0.11, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5]
for ratio in ratios:
    compressed_model = compression_fn(
        model,
        ratio_to_keep=ratio,
        inplace=False,
        model_initializer=lambda: copy.deepcopy(model),
        should_do=should_compress,
    )
    compressed_model.to(device)

    loss, accuracy = evaluate(compressed_model, test_loader, device)
    n_params = sum(p.numel() for p in compressed_model.parameters())
    print(
        f"Ratio: {ratio:.2f}, Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}, Number of Parameters: {n_params}, Compression Ratio: {n_params / nparams_orig:.2f}"
    )

    if args.print_model:
        print(compressed_model)
