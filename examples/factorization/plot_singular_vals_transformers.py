from compress.factorization.factorize import plot_singular_values
import argparse
from transformers import AutoModelForSequenceClassification

parser = argparse.ArgumentParser(description="Plot singular values of a model")
parser.add_argument(
    "--model_name", type=str, default="textattack/bert-base-uncased-imdb"
)
args = parser.parse_args()

model = AutoModelForSequenceClassification.from_pretrained(args.model_name)

plot_singular_values(model)
