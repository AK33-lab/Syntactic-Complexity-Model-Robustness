from data_loader import load_data
from models import DEVICE, set_seed, MLPClassifier, RNNClassifier, train_model, load_roberta, load_bart
from evaluate import evaluate

set_seed(67)
data = load_data()

print(f"Using device: {DEVICE}")

print("Training MLP...")
mlp = MLPClassifier()
train_model(mlp, data, "mlp_weights.pt")

print("Training RNN...")
rnn = RNNClassifier()
train_model(rnn, data, "rnn_weights.pt")

print("Loading RoBERTa...")
roberta_tokenizer, roberta_model = load_roberta()
print("Loading BART...")
bart_tokenizer, bart_model = load_bart()

baseline_results = evaluate(data, mlp, rnn, roberta_tokenizer, roberta_model, bart_tokenizer, bart_model)