import argparse

from data_loader import load_data
from models import DEVICE, RNNClassifier, set_seed, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNN on original MNLI examples.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--weights-path",
        default="rnn_weights.pt",
        help="Output checkpoint path for RNN weights.",
    )
    parser.add_argument(
        "--data-path",
        default="data.jsonl",
        help="Path to expanded JSONL with perturbation rows.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(67)
    print(f"Using device: {DEVICE}")

    data = load_data(save_path=args.data_path)
    original_data = [ex for ex in data if ex.get("perturbation_method") == "original"]
    print(f"Training RNN on {len(original_data)} original examples")

    model = RNNClassifier().to(DEVICE)
    losses = train_model(
        model,
        original_data,
        args.weights_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print("\n=== RNN Training Summary ===")
    for epoch, loss in enumerate(losses, 1):
        print(f"Epoch {epoch}: {loss:.4f}")


if __name__ == "__main__":
    main()
