import os
import torch
from data_loader import load_data
from models import (
    DEVICE,
    set_seed,
    MLPClassifier,
    RNNClassifier,
    train_model,
)

set_seed(67)
print(f"Using device: {DEVICE}")

# Load only original examples for training
data = load_data()
original_data = [ex for ex in data if ex["perturbation_method"] == "original"]
print(f"Training on {len(original_data)} original examples")

# Train MLP
print("Training MLP...")
mlp = MLPClassifier().to(DEVICE)
mlp_losses = train_model(mlp, original_data, "mlp_weights.pt", epochs=3)

# Train RNN
print("Training RNN...")
rnn = RNNClassifier().to(DEVICE)
rnn_losses = train_model(rnn, original_data, "rnn_weights.pt", epochs=3)

print("\n=== Training Summary ===")
print(f"{'Epoch':<8} {'MLP Loss':<12} {'RNN Loss':<12}")
for epoch, (ml, rl) in enumerate(zip(mlp_losses, rnn_losses), 1):
    print(f"{epoch:<8} {ml:<12.4f} {rl:<12.4f}")
print("Done!")