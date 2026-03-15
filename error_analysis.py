import argparse
import csv
import os
import random
from collections import defaultdict

import torch

from data_loader import DEFAULT_PERTURBATIONS, load_data
from evaluate import evaluate
from models import (
    DEVICE,
    MLPClassifier,
    RNNClassifier,
    load_bart,
    load_roberta,
    set_seed,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze cases where models get original correct but perturbed wrong."
    )
    parser.add_argument(
        "--data-path",
        default=os.path.join(SCRIPT_DIR, "data.jsonl"),
        help="Path to expanded JSONL containing original and perturbed rows.",
    )
    parser.add_argument(
        "--mlp-weights",
        default=os.path.join(SCRIPT_DIR, "mlp_weights.pt"),
        help="Path to saved MLP checkpoint.",
    )
    parser.add_argument(
        "--rnn-weights",
        default=os.path.join(SCRIPT_DIR, "rnn_weights.pt"),
        help="Path to saved RNN checkpoint.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mlp", "rnn", "roberta", "bart"],
        default=["mlp", "rnn", "roberta", "bart"],
        help="Models to include in error analysis.",
    )
    parser.add_argument(
        "--perturb",
        nargs="+",
        choices=["adjrel", "passive", "pp"],
        default=list(DEFAULT_PERTURBATIONS),
        help="Perturbation sets to analyze.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=2000,
        help=(
            "Maximum number of original example_ids to analyze (shared across all methods). "
            "Use 0 or a negative value to analyze all examples."
        ),
    )
    parser.add_argument(
        "--cases-csv",
        default=os.path.join(SCRIPT_DIR, "error_cases.csv"),
        help="Output CSV for per-example failure cases.",
    )
    parser.add_argument(
        "--summary-csv",
        default=os.path.join(SCRIPT_DIR, "error_summary.csv"),
        help="Output CSV for matched/mismatched failure summaries.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=2,
        help="How many example pairs to print per (model, perturbation, split).",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def group_by_perturbation(data):
    grouped = defaultdict(list)
    for ex in data:
        method = ex.get("perturbation_method", "original")
        grouped[method].append(ex)
    return grouped


def write_csv(path, fieldnames, rows):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    models_to_eval = list(dict.fromkeys(args.models))
    perturbations = list(dict.fromkeys(args.perturb))

    set_seed(67)
    if torch.cuda.is_available():
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {DEVICE}")

    data = load_data(save_path=args.data_path, perturbations=DEFAULT_PERTURBATIONS)
    grouped = group_by_perturbation(data)

    if "original" not in grouped:
        raise ValueError("Missing 'original' rows in loaded data.")

    missing_methods = [m for m in perturbations if m not in grouped]
    if missing_methods:
        raise ValueError(
            "Missing perturbation methods in loaded data: " + ", ".join(missing_methods)
        )

    original_all = grouped["original"]
    all_ids = [ex["example_id"] for ex in original_all]
    if args.max_examples > 0 and args.max_examples < len(all_ids):
        rng = random.Random(67)
        sampled_ids = set(rng.sample(all_ids, args.max_examples))
        print(
            f"Sampling {len(sampled_ids)} / {len(all_ids)} example_ids for error analysis."
        )
    else:
        sampled_ids = set(all_ids)
        print(f"Using all {len(sampled_ids)} example_ids for error analysis.")

    mlp_model = None
    rnn_model = None
    roberta_tokenizer = None
    roberta_model = None
    bart_tokenizer = None
    bart_model = None

    if "mlp" in models_to_eval:
        print("Loading saved MLP checkpoint...")
        mlp_model = load_checkpoint(MLPClassifier(), args.mlp_weights)

    if "rnn" in models_to_eval:
        print("Loading saved RNN checkpoint...")
        rnn_model = load_checkpoint(RNNClassifier(), args.rnn_weights)

    if "roberta" in models_to_eval:
        print("Loading RoBERTa...")
        roberta_tokenizer, roberta_model = load_roberta()

    if "bart" in models_to_eval:
        print("Loading BART...")
        bart_tokenizer, bart_model = load_bart()

    original_subset = [ex for ex in grouped["original"] if ex["example_id"] in sampled_ids]
    print(f"\nEvaluating original set ({len(original_subset)} examples)...")
    original_results = evaluate(
        original_subset,
        mlp=mlp_model,
        rnn=rnn_model,
        roberta_tokenizer=roberta_tokenizer,
        roberta_model=roberta_model,
        bart_tokenizer=bart_tokenizer,
        bart_model=bart_model,
        models_to_eval=models_to_eval,
        batch_size=args.batch_size,
    )

    original_by_id = {ex["example_id"]: ex for ex in original_subset}
    original_correct = {
        model_name: {
            ex["example_id"]: bool(ok)
            for ex, ok in zip(original_subset, original_results[model_name])
        }
        for model_name in models_to_eval
    }

    case_rows = []
    summary_rows = []
    sample_groups = defaultdict(list)

    split_order = {"matched": 0, "mismatched": 1, "overall": 2}

    for perturb in perturbations:
        subset = [ex for ex in grouped[perturb] if ex["example_id"] in sampled_ids]
        print(f"\nEvaluating perturbation '{perturb}' ({len(subset)} examples)...")
        perturb_results = evaluate(
            subset,
            mlp=mlp_model,
            rnn=rnn_model,
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            bart_tokenizer=bart_tokenizer,
            bart_model=bart_model,
            models_to_eval=models_to_eval,
            batch_size=args.batch_size,
        )

        for model_name in models_to_eval:
            split_stats = {
                "matched": {"original_correct": 0, "perturbed_failures": 0},
                "mismatched": {"original_correct": 0, "perturbed_failures": 0},
            }

            for ex, perturb_ok in zip(subset, perturb_results[model_name]):
                ex_id = ex["example_id"]
                split = ex.get("source_split", "unknown")
                if split not in split_stats:
                    split_stats[split] = {"original_correct": 0, "perturbed_failures": 0}

                orig_ok = original_correct[model_name].get(ex_id, False)
                if not orig_ok:
                    continue

                split_stats[split]["original_correct"] += 1
                if perturb_ok:
                    continue

                split_stats[split]["perturbed_failures"] += 1
                original_ex = original_by_id.get(ex_id, ex)
                case_rows.append(
                    {
                        "model": model_name,
                        "perturbation method": perturb,
                        "source split": split,
                        "example_id": ex_id,
                        "label": ex["label"],
                        "hypothesis": ex["hypothesis"],
                        "original premise": original_ex["premise"],
                        "perturbed premise": ex["premise"],
                    }
                )

                group_key = (model_name, perturb, split)
                if len(sample_groups[group_key]) < args.samples_per_group:
                    sample_groups[group_key].append(
                        {
                            "example_id": ex_id,
                            "label": ex["label"],
                            "hypothesis": ex["hypothesis"],
                            "original premise": original_ex["premise"],
                            "perturbed premise": ex["premise"],
                        }
                    )

            overall_original_correct = 0
            overall_perturbed_failures = 0
            for split, stats in split_stats.items():
                orig_correct_count = stats["original_correct"]
                failure_count = stats["perturbed_failures"]
                failure_rate = (
                    failure_count / orig_correct_count if orig_correct_count else 0.0
                )
                summary_rows.append(
                    {
                        "model": model_name,
                        "perturbation method": perturb,
                        "source split": split,
                        "original correct": orig_correct_count,
                        "perturbed failures": failure_count,
                        "failure rate given original correct": f"{failure_rate:.6f}",
                    }
                )
                overall_original_correct += orig_correct_count
                overall_perturbed_failures += failure_count

            overall_rate = (
                overall_perturbed_failures / overall_original_correct
                if overall_original_correct
                else 0.0
            )
            summary_rows.append(
                {
                    "model": model_name,
                    "perturbation method": perturb,
                    "source split": "overall",
                    "original correct": overall_original_correct,
                    "perturbed failures": overall_perturbed_failures,
                    "failure rate given original correct": f"{overall_rate:.6f}",
                }
            )

    case_rows.sort(
        key=lambda r: (
            r["model"],
            r["perturbation method"],
            split_order.get(r["source split"], 99),
            int(r["example_id"]),
        )
    )
    summary_rows.sort(
        key=lambda r: (
            r["model"],
            r["perturbation method"],
            split_order.get(r["source split"], 99),
        )
    )

    write_csv(
        args.cases_csv,
        [
            "model",
            "perturbation method",
            "source split",
            "example_id",
            "label",
            "hypothesis",
            "original premise",
            "perturbed premise",
        ],
        case_rows,
    )
    write_csv(
        args.summary_csv,
        [
            "model",
            "perturbation method",
            "source split",
            "original correct",
            "perturbed failures",
            "failure rate given original correct",
        ],
        summary_rows,
    )

    print("\n=== Error Summary (overall rows) ===")
    print(
        f"{'model':<10} {'perturbation':<10} {'orig_correct':<13} "
        f"{'flip_fail':<10} {'flip_rate':<10}"
    )
    for row in summary_rows:
        if row["source split"] != "overall":
            continue
        print(
            f"{row['model']:<10} {row['perturbation method']:<10} "
            f"{row['original correct']:<13} {row['perturbed failures']:<10} "
            f"{row['failure rate given original correct']:<10}"
        )

    if args.samples_per_group > 0:
        print("\n=== Sample Failure Pairs (original correct -> perturbed wrong) ===")
        for key in sorted(sample_groups.keys()):
            model_name, perturb, split = key
            examples = sample_groups[key]
            if not examples:
                continue
            print(f"\n[{model_name} | {perturb} | {split}] {len(examples)} sample(s)")
            for idx, sample in enumerate(examples, 1):
                print(f"{idx}. example_id={sample['example_id']} label={sample['label']}")
                print(f"   hypothesis: {sample['hypothesis']}")
                print(f"   original:  {sample['original premise']}")
                print(f"   perturbed: {sample['perturbed premise']}")

    print(f"\nSaved failure cases: {args.cases_csv}")
    print(f"Saved error summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
