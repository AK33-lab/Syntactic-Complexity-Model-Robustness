import argparse
import os

import torch

from data_loader import load_data
from evaluate import evaluate
from models import (
    DEVICE,
    MLPClassifier,
    RNNClassifier,
    load_bart,
    load_roberta,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate baseline accuracy on original data.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--data-path",
        default=os.path.join(script_dir, "data.jsonl"),
        help="Path to expanded JSONL containing original and perturbed rows.",
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
        "--models",
        nargs="+",
        choices=["mlp", "rnn", "roberta", "bart"],
        default=["mlp", "rnn"],
        help="Models to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def accuracy_from_preds(preds):
    if not preds:
        return 0.0
    return sum(preds) / len(preds)


def main():
    args = parse_args()
    models_to_eval = list(dict.fromkeys(args.models))

    set_seed(67)
    if torch.cuda.is_available():
        print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {DEVICE}")

    data = load_data(save_path=args.data_path)
    baseline_subset = [ex for ex in data if ex.get("perturbation_method") == "original"]
    if not baseline_subset:
        raise ValueError("No original examples found in loaded data.")

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

    print(f"\nEvaluating baseline on original data ({len(baseline_subset)} examples)")
    results = evaluate(
        baseline_subset,
        mlp=mlp_model,
        rnn=rnn_model,
        roberta_tokenizer=roberta_tokenizer,
        roberta_model=roberta_model,
        bart_tokenizer=bart_tokenizer,
        bart_model=bart_model,
        models_to_eval=models_to_eval,
        batch_size=args.batch_size,
    )

    print("\n=== Baseline Accuracy (original) ===")
    for model_name in models_to_eval:
        acc = accuracy_from_preds(results[model_name])
        print(f"{model_name}: {acc:.6f}")


if __name__ == "__main__":
    main()
