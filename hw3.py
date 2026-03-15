import argparse
import csv
import os
from collections import defaultdict

import torch
from data_loader import DEFAULT_PERTURBATIONS, load_data
from evaluate import evaluate
from metrics import compute_complexity
from models import (
    DEVICE,
    set_seed,
    MLPClassifier,
    RNNClassifier,
    load_roberta,
    load_bart,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved MNLI models on perturbed premises."
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--perturb",
        nargs="+",
        choices=["adjrel", "passive", "pp"],
        default=list(DEFAULT_PERTURBATIONS),
        help="Perturbation subsets to report in output files.",
    )
    parser.add_argument(
        "--data-path",
        default=os.path.join(script_dir, "data.jsonl"),
        help="Path to expanded JSONL containing original and perturbed rows.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild data.jsonl from MNLI and perturbation functions.",
    )
    parser.add_argument(
        "--mlp-weights",
        default=os.path.join(script_dir, "mlp_weights.pt"),
        help="Path to saved MLP checkpoint.",
    )
    parser.add_argument(
        "--rnn-weights",
        default=os.path.join(script_dir, "rnn_weights.pt"),
        help="Path to saved RNN checkpoint.",
    )
    parser.add_argument(
        "--perf-csv",
        default=os.path.join(script_dir, "perf.csv"),
        help="Output CSV path with model performance by perturbation method.",
    )
    parser.add_argument(
        "--complex-csv",
        default=os.path.join(script_dir, "complex.csv"),
        help="Output CSV path with complexity metrics by perturbation method.",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def group_data_by_perturbation(data):
    grouped = defaultdict(list)
    for ex in data:
        method = ex.get("perturbation_method", "original")
        grouped[method].append(ex)
    return grouped


def compute_average_complexity(data):
    totals = defaultdict(float)
    counts = defaultdict(int)

    for ex in data:
        metrics = compute_complexity(ex["premise"])
        for metric_type, value in metrics.items():
            totals[metric_type] += float(value)
            counts[metric_type] += 1

    averages = {}
    for metric_type, total in totals.items():
        averages[metric_type] = total / counts[metric_type]
    return averages


def write_csv(path, fieldnames, rows):
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def accuracy_from_preds(preds):
    if not preds:
        return 0.0
    return sum(preds) / len(preds)


def main():
    args = parse_args()
    eval_targets = ["mlp", "rnn", "roberta", "bart"]
    baseline_method = "original"
    selected_perturbations = list(dict.fromkeys(args.perturb))
    methods_to_report = selected_perturbations

    set_seed(67)
    if torch.cuda.is_available():
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {DEVICE}")

    data = load_data(
        save_path=args.data_path,
        rebuild=args.rebuild_data,
        perturbations=DEFAULT_PERTURBATIONS,
    )
    grouped_data = group_data_by_perturbation(data)

    missing_methods = [m for m in methods_to_report if m not in grouped_data]
    if missing_methods:
        raise ValueError(
            "Missing perturbation methods in loaded data: " + ", ".join(missing_methods)
        )
    if baseline_method not in grouped_data:
        raise ValueError("Missing baseline method in loaded data: original")

    print("Loading saved MLP checkpoint...")
    mlp_model = load_checkpoint(MLPClassifier(), args.mlp_weights)
    print(f"MLP loaded on: {next(mlp_model.parameters()).device}")

    print("Loading saved RNN checkpoint...")
    rnn_model = load_checkpoint(RNNClassifier(), args.rnn_weights)
    print(f"RNN loaded on: {next(rnn_model.parameters()).device}")

    print("Loading RoBERTa...")
    roberta_tokenizer, roberta_model = load_roberta()
    print(f"RoBERTa loaded on: {roberta_model.device}")

    print("Loading BART...")
    bart_tokenizer, bart_model = load_bart()
    print(f"BART loaded on: {bart_model.device}")

    perf_rows = []
    complex_rows = []

    baseline_subset = grouped_data[baseline_method]
    print(f"\nBaseline evaluation on original data ({len(baseline_subset)} examples)")
    evaluate(
        baseline_subset,
        mlp=mlp_model,
        rnn=rnn_model,
        roberta_tokenizer=roberta_tokenizer,
        roberta_model=roberta_model,
        bart_tokenizer=bart_tokenizer,
        bart_model=bart_model,
        models_to_eval=eval_targets,
    )

    for perturb_name in methods_to_report:
        subset = grouped_data[perturb_name]

        print(f"\nEvaluating models on perturbation: {perturb_name} ({len(subset)} examples)")
        model_results = evaluate(
            subset,
            mlp=mlp_model,
            rnn=rnn_model,
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            bart_tokenizer=bart_tokenizer,
            bart_model=bart_model,
            models_to_eval=eval_targets,
        )

        for model_name in eval_targets:
            performance = accuracy_from_preds(model_results[model_name])
            perf_rows.append(
                {
                    "model": model_name,
                    "perturbation method": perturb_name,
                    "performance": f"{performance:.6f}",
                }
            )

        complexity_scores = compute_average_complexity(subset)
        for metric_type, value in complexity_scores.items():
            complex_rows.append(
                {
                    "perturbation method": perturb_name,
                    "metric type": metric_type,
                    "value": f"{value:.6f}",
                }
            )

    perf_rows.sort(key=lambda row: (row["perturbation method"], row["model"]))
    complex_rows.sort(key=lambda row: (row["perturbation method"], row["metric type"]))

    write_csv(
        args.perf_csv,
        ["model", "perturbation method", "performance"],
        perf_rows,
    )
    write_csv(
        args.complex_csv,
        ["perturbation method", "metric type", "value"],
        complex_rows,
    )

    print(f"Saved performance summary to: {args.perf_csv}")
    print(f"Saved complexity summary to: {args.complex_csv}")

if __name__ == "__main__":
    main()