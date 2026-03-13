from tqdm import tqdm
from models import predict_mlp, predict_rnn, predict_transformer

def evaluate(data, mlp, rnn, roberta_tokenizer, roberta_model, bart_tokenizer, bart_model):
    results = {"mlp": [], "rnn": [], "roberta": [], "bart": []}
    for ex in tqdm(data, desc="Evaluating"):
        p, h, label = ex["premise"], ex["hypothesis"], ex["label"]
        results["mlp"].append(predict_mlp(mlp, p, h) == label)
        results["rnn"].append(predict_rnn(rnn, p, h) == label)
        results["roberta"].append(predict_transformer(roberta_tokenizer, roberta_model, p, h) == label)
        results["bart"].append(predict_transformer(bart_tokenizer, bart_model, p, h) == label)
    for model_name, preds in results.items():
        acc = sum(preds) / len(preds)
        print(f"{model_name} accuracy: {acc:.4f}")
    return results