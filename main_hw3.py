import csv
import os
from collections import defaultdict

import torch

from data_loader import DEFAULT_PERTURBATIONS, load_data
from evaluate import evaluate
from metrics import compute_complexity
from models import (
    DEVICE,
    MLPClassifier,
    RNNClassifier,
    load_bart,
    load_roberta,
    set_seed,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_TO_EVAL = ["mlp", "rnn", "roberta", "bart"]
PERTURBATIONS = list(DEFAULT_PERTURBATIONS)  # ["adjrel", "passive", "pp"]
ALL_METHODS = ["original"] + PERTURBATIONS

DATA_PATH = os.path.join(SCRIPT_DIR, "data.jsonl")
MLP_WEIGHTS = os.path.join(SCRIPT_DIR, "mlp_weights.pt")
RNN_WEIGHTS = os.path.join(SCRIPT_DIR, "rnn_weights.pt")
PERF_CSV = os.path.join(SCRIPT_DIR, "perf.csv")
COMPLEX_CSV = os.path.join(SCRIPT_DIR, "complex.csv")


def load_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)


def group_by_perturbation(data):
    grouped = defaultdict(list)
    for ex in data:
        grouped[ex.get("perturbation_method", "original")].append(ex)
    return grouped


def compute_average_complexity(data):
    totals = defaultdict(float)
    counts = defaultdict(int)
    for ex in data:
        metrics = compute_complexity(ex["premise"])
        if metrics is None:
            continue
        for metric_type, value in metrics.items():
            totals[metric_type] += float(value)
            counts[metric_type] += 1
    return {k: totals[k] / counts[k] for k in totals}


def accuracy(preds):
    return sum(preds) / len(preds) if preds else 0.0


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(67)
    print(f"Using device: {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────────
    data = load_data(save_path=DATA_PATH, perturbations=DEFAULT_PERTURBATIONS)
    grouped = group_by_perturbation(data)

    for method in ALL_METHODS:
        if method not in grouped:
            raise ValueError(f"Missing perturbation method in data: {method}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading MLP...")
    mlp_model = load_checkpoint(MLPClassifier(), MLP_WEIGHTS)

    print("Loading RNN...")
    rnn_model = load_checkpoint(RNNClassifier(), RNN_WEIGHTS)

    print("Loading RoBERTa...")
    roberta_tokenizer, roberta_model = load_roberta()

    print("Loading BART...")
    bart_tokenizer, bart_model = load_bart()

    # ── Evaluate each method ───────────────────────────────────────────────────
    perf_rows = []
    complex_rows = []

    for method in ALL_METHODS:
        subset = grouped[method]
        print(f"\nEvaluating '{method}' ({len(subset)} examples)...")

        results = evaluate(
            subset,
            mlp=mlp_model,
            rnn=rnn_model,
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            bart_tokenizer=bart_tokenizer,
            bart_model=bart_model,
            models_to_eval=MODELS_TO_EVAL,
        )

        for model_name in MODELS_TO_EVAL:
            perf_rows.append({
                "model": model_name,
                "perturbation method": method,
                "performance": f"{accuracy(results[model_name]):.6f}",
            })

        for metric_type, value in compute_average_complexity(subset).items():
            complex_rows.append({
                "perturbation method": method,
                "metric type": metric_type,
                "value": f"{value:.6f}",
            })

    # ── Sort and save CSVs ─────────────────────────────────────────────────────
    perf_rows.sort(key=lambda r: (r["perturbation method"], r["model"]))
    complex_rows.sort(key=lambda r: (r["perturbation method"], r["metric type"]))

    write_csv(PERF_CSV, ["model", "perturbation method", "performance"], perf_rows)
    write_csv(COMPLEX_CSV, ["perturbation method", "metric type", "value"], complex_rows)

    print(f"\nSaved: {PERF_CSV}")
    print(f"Saved: {COMPLEX_CSV}")


if __name__ == "__main__":
    main()
