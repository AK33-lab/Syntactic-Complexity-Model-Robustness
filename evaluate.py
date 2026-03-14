from tqdm import tqdm
from models import predict_mlp, predict_rnn, predict_transformer

LABEL_MAP = {0: 2, 1: 1, 2: 0}

def evaluate(
    data,
    mlp=None,
    rnn=None,
    roberta_tokenizer=None,
    roberta_model=None,
    bart_tokenizer=None,
    bart_model=None,
    models_to_eval=None,
):
    if models_to_eval is None:
        models_to_eval = ["mlp", "rnn", "roberta", "bart"]

    results = {model_name: [] for model_name in models_to_eval}

    if "mlp" in models_to_eval and mlp is None:
        raise ValueError("MLP model is required when evaluating 'mlp'.")
    if "rnn" in models_to_eval and rnn is None:
        raise ValueError("RNN model is required when evaluating 'rnn'.")
    if "roberta" in models_to_eval and (roberta_tokenizer is None or roberta_model is None):
        raise ValueError("RoBERTa tokenizer/model are required when evaluating 'roberta'.")
    if "bart" in models_to_eval and (bart_tokenizer is None or bart_model is None):
        raise ValueError("BART tokenizer/model are required when evaluating 'bart'.")

    for ex in tqdm(data, desc="Evaluating"):
        p, h, label = ex["premise"], ex["hypothesis"], ex["label"]

        if "mlp" in results:
            results["mlp"].append(predict_mlp(mlp, p, h) == label)
        if "rnn" in results:
            results["rnn"].append(predict_rnn(rnn, p, h) == label)
        if "roberta" in results:
            roberta_pred = predict_transformer(roberta_tokenizer, roberta_model, p, h)
            roberta_pred = LABEL_MAP.get(roberta_pred, roberta_pred)
            results["roberta"].append(
                roberta_pred == label
            )
        if "bart" in results:
            bart_pred = predict_transformer(bart_tokenizer, bart_model, p, h)
            bart_pred = LABEL_MAP.get(bart_pred, bart_pred)
            results["bart"].append(
                bart_pred == label
            )

    for model_name, preds in results.items():
        acc = sum(preds) / len(preds)
        print(f"{model_name} accuracy: {acc:.4f}")

    return results