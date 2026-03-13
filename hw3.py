from data_loader import load_data
from models import set_seed, MLPClassifier, RNNClassifier, train_model, load_roberta, load_bart
from evaluate import evaluate

set_seed(67)
data = load_data()

print("Training MLP...")
mlp = MLPClassifier()
train_model(mlp, data, "mlp_weights.pt", epochs=1)

print("Training RNN...")
rnn = RNNClassifier()
train_model(rnn, data, "rnn_weights.pt", epochs=1)

print("Loading RoBERTa...")
roberta_tokenizer, roberta_model = load_roberta()
print("Loading BART...")
bart_tokenizer, bart_model = load_bart()

baseline_results = evaluate(data, mlp, rnn, roberta_tokenizer, roberta_model, bart_tokenizer, bart_model)