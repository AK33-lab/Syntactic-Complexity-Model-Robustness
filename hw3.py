import argparse
import os

import torch

from data_loader import load_data
from models import (
    DEVICE,
    set_seed,
    MLPClassifier,
    RNNClassifier,
    train_model,
    load_roberta,
    load_bart,
)
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MLP/RNN and evaluate MLP/RNN/RoBERTa/BART."
    )
    parser.add_argument(
        "--train",
        nargs="+",
        choices=["mlp", "rnn"],
        default=[],
        help="Models to train.",
    )
    parser.add_argument(
        "--eval",
        nargs="+",
        choices=["mlp", "rnn", "roberta", "bart"],
        default=[],
        help="Models to evaluate (RoBERTa/BART are pretrained).",
    )
    parser.add_argument("--mlp-weights", default="mlp_weights.pt", help="Path to save/load MLP weights.")
    parser.add_argument("--rnn-weights", default="rnn_weights.pt", help="Path to save/load RNN weights.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)


def main():
    args = parse_args()
    eval_targets = list(dict.fromkeys(args.eval))

    set_seed(67)
    print(f"Using device: {DEVICE}")

    if not args.train and not eval_targets:
        print("No action requested. Use --train (mlp/rnn) and/or --eval (mlp/rnn/roberta/bart).")
        return

    data = load_data()
    trained_models = {}

    if "mlp" in args.train:
        print("Training MLP...")
        mlp = MLPClassifier()
        train_model(
            mlp,
            data,
            args.mlp_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        trained_models["mlp"] = mlp

    if "rnn" in args.train:
        print("Training RNN...")
        rnn = RNNClassifier()
        train_model(
            rnn,
            data,
            args.rnn_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        trained_models["rnn"] = rnn

    eval_models = {}

    if "mlp" in eval_targets:
        if "mlp" in trained_models:
            eval_models["mlp"] = trained_models["mlp"]
        else:
            print(f"Loading MLP weights from {args.mlp_weights}...")
            eval_models["mlp"] = load_checkpoint(MLPClassifier(), args.mlp_weights)

    if "rnn" in eval_targets:
        if "rnn" in trained_models:
            eval_models["rnn"] = trained_models["rnn"]
        else:
            print(f"Loading RNN weights from {args.rnn_weights}...")
            eval_models["rnn"] = load_checkpoint(RNNClassifier(), args.rnn_weights)

    roberta_tokenizer, roberta_model = None, None
    bart_tokenizer, bart_model = None, None

    if "roberta" in eval_targets:
        print("Loading RoBERTa...")
        roberta_tokenizer, roberta_model = load_roberta()

    if "bart" in eval_targets:
        print("Loading BART...")
        bart_tokenizer, bart_model = load_bart()

    if eval_targets:
        evaluate(
            data,
            mlp=eval_models.get("mlp"),
            rnn=eval_models.get("rnn"),
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            bart_tokenizer=bart_tokenizer,
            bart_model=bart_model,
            models_to_eval=eval_targets,
        )


if __name__ == "__main__":
    main()