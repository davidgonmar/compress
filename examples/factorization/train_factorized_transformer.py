import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler,
)
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import argparse

from compress.regularizers import (
    SingularValuesRegularizer,
    extract_weights_and_reshapers,
    update_weights,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="textattack/bert-base-uncased-imdb"
)
parser.add_argument("--dataset_name", type=str, default="imdb")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--save_path", type=str, default="bert_imdb")
parser.add_argument("--sv_regularizer", type=str, default="noop")
parser.add_argument("--regularizer_weight", type=float, default=1.0)
parser.add_argument("--regularizer_scheduler", type=str, default="noop")
parser.add_argument("--finetune", action="store_true")
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
).shuffle()
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


dataset["train"] = dataset["train"].select(range(2000))
dataset["test"] = dataset["test"].select(range(50))
train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(dataset["test"], batch_size=args.batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * args.epochs,
)

regularizer_kwargs = {
    "entropy": {},
    "hoyer_sparsity": {"normalize": True},
    "scad": {"lambda_val": 0.1, "a_val": 3.7},
    "squared_hoyer_sparsity": {"normalize": True},
    "nuclear_norm": {},
    "approximated_hoyer_sparsity": {"normalize": True},
    "noop": {},
}
weights = args.regularizer_weight


def weight_schedule_noop(epochnum):
    return weights


def weight_schedule_exp(epochnum):
    return 0.5 * weights * 2 ** ((epochnum / args.epochs))


weight_sched = {
    "noop": weight_schedule_noop,
    "exp": weight_schedule_exp,
}[args.regularizer_scheduler]

regularizer = SingularValuesRegularizer(
    metric=args.sv_regularizer,
    params_and_reshapers=extract_weights_and_reshapers(
        model,
        cls_list=(torch.nn.Linear,),
        keywords={"weight"},
    ),
    weights=args.regularizer_weight,
    **regularizer_kwargs[args.sv_regularizer],
)

num_epochs = args.epochs
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    reg_loss = 0.0
    update_weights(regularizer, weight_sched(epoch))

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        reg = regularizer()
        total_loss = loss + reg

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * input_ids.size(0)
        reg_loss += reg.item() * input_ids.size(0)

    train_loss /= len(train_loader.dataset)
    reg_loss /= len(train_loader.dataset)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Regularization Loss: {reg_loss:.4f}"
    )

    model.eval()
    val_loss = 0.0
    metric = load_metric("accuracy")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(outputs.logits, dim=1)
            metric.add_batch(predictions=preds, references=labels)

    val_loss /= len(val_loader.dataset)
    accuracy = metric.compute()["accuracy"]
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
