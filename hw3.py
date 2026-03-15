import argparse
import os
import torch
from data_loader import load_data
from models import (
    DEVICE, set_seed, MLPClassifier, RNNClassifier,
    train_model, load_roberta, load_bart,
)
from evaluate import evaluate
from perturbations import adj_to_relative_clause, passive_voice, prepositional_phrase_insertion

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", choices=["mlp", "rnn"], default=[])
    parser.add_argument("--eval", nargs="+", choices=["mlp", "rnn", "roberta", "bart"], default=[])
    parser.add_argument("--perturb", nargs="+", choices=["adjrel", "passive", "pp"], default=[],
                        help="Perturbations to apply and evaluate.")
    parser.add_argument("--mlp-weights", default="mlp_weights.pt")
    parser.add_argument("--rnn-weights", default="rnn_weights.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    return model.to(DEVICE)

def apply_perturbation(data, perturb_fn):
    perturbed = []
    for ex in data:
        new_ex = dict(ex)
        new_ex["premise"] = perturb_fn(ex["premise"])
        perturbed.append(new_ex)
    return perturbed

def main():
    args = parse_args()
    eval_targets = list(dict.fromkeys(args.eval))
    set_seed(67)
    print(f"Using device: {DEVICE}")

    if not args.train and not eval_targets and not args.perturb:
        print("No action requested.")
        return

    data = load_data()
    trained_models = {}

    # Training
    if "mlp" in args.train:
        print("Training MLP...")
        mlp = MLPClassifier()
        train_model(mlp, data, args.mlp_weights, epochs=args.epochs, batch_size=args.batch_size)
        trained_models["mlp"] = mlp

    if "rnn" in args.train:
        print("Training RNN...")
        rnn = RNNClassifier()
        train_model(rnn, data, args.rnn_weights, epochs=args.epochs, batch_size=args.batch_size)
        trained_models["rnn"] = rnn

    # Load eval models
    eval_models = {}
    if "mlp" in eval_targets:
        eval_models["mlp"] = trained_models.get("mlp") or load_checkpoint(MLPClassifier(), args.mlp_weights)
    if "rnn" in eval_targets:
        eval_models["rnn"] = trained_models.get("rnn") or load_checkpoint(RNNClassifier(), args.rnn_weights)

    roberta_tokenizer, roberta_model = None, None
    bart_tokenizer, bart_model = None, None
    if "roberta" in eval_targets or args.perturb:
        print("Loading RoBERTa...")
        roberta_tokenizer, roberta_model = load_roberta()
    if "bart" in eval_targets or args.perturb:
        print("Loading BART...")
        bart_tokenizer, bart_model = load_bart()

    # Baseline evaluation
    if eval_targets:
        print("Baseline evaluation...")
        evaluate(data, mlp=eval_models.get("mlp"), rnn=eval_models.get("rnn"),
                 roberta_tokenizer=roberta_tokenizer, roberta_model=roberta_model,
                 bart_tokenizer=bart_tokenizer, bart_model=bart_model,
                 models_to_eval=eval_targets)

    # Perturbation evaluation
    perturb_map = {
        "adjrel": adj_to_relative_clause,
        "passive": passive_voice,
        "pp": prepositional_phrase_insertion
    }
    for perturb_name in args.perturb:
        print(f"\nApplying perturbation: {perturb_name}")
        perturbed_data = apply_perturbation(data, perturb_map[perturb_name])
        evaluate(perturbed_data, mlp=eval_models.get("mlp"), rnn=eval_models.get("rnn"),
                 roberta_tokenizer=roberta_tokenizer, roberta_model=roberta_model,
                 bart_tokenizer=bart_tokenizer, bart_model=bart_model,
                 models_to_eval=eval_targets, perturbation=perturb_name)

if __name__ == "__main__":
    main()