from tqdm import tqdm
import torch
from models import DEVICE

LABEL_MAP = {0: 2, 1: 1, 2: 0}

def evaluate(data, mlp=None, rnn=None, roberta_tokenizer=None, roberta_model=None,
             bart_tokenizer=None, bart_model=None, models_to_eval=None, batch_size=32):
    if models_to_eval is None:
        models_to_eval = ["mlp", "rnn", "roberta", "bart"]

    results = {model_name: [] for model_name in models_to_eval}
    labels = [ex["label"] for ex in data]

    # Batch MLP and RNN
    if "mlp" in models_to_eval or "rnn" in models_to_eval:
        from models import get_embedding
        print("Computing embeddings...")
        embeddings = []
        for ex in tqdm(data, desc="Embedding"):
            emb = get_embedding(ex["premise"], ex["hypothesis"])
            embeddings.append(emb.squeeze(0))

        for model_name, model in [("mlp", mlp), ("rnn", rnn)]:
            if model_name not in models_to_eval:
                continue
            print(f"Evaluating {model_name}...")
            model.eval()
            for i in tqdm(range(0, len(embeddings), batch_size)):
                # DEBUG
                if i == 0:
                    print(f"Type of embeddings[0]: {type(embeddings[0])}")
                    print(f"Length of embeddings: {len(embeddings)}")
                batch_embs = torch.stack(embeddings[i:i+batch_size]).to(DEVICE)
                batch_labels = labels[i:i+batch_size]
                with torch.no_grad():
                    preds = torch.argmax(model(batch_embs), dim=1).tolist()
                results[model_name].extend([p == l for p, l in zip(preds, batch_labels)])

    # Batch RoBERTa and BART
    for model_name, tokenizer, model in [
        ("roberta", roberta_tokenizer, roberta_model),
        ("bart", bart_tokenizer, bart_model)
    ]:
        if model_name not in models_to_eval:
            continue
        print(f"Evaluating {model_name}...")
        model.eval()
        for i in tqdm(range(0, len(data), batch_size)):
            batch = data[i:i+batch_size]
            premises = [ex["premise"] for ex in batch]
            hypotheses = [ex["hypothesis"] for ex in batch]
            batch_labels = labels[i:i+batch_size]
            inputs = tokenizer(
                premises, hypotheses,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).tolist()
                preds = [LABEL_MAP.get(p, p) for p in preds]
            results[model_name].extend([p == l for p, l in zip(preds, batch_labels)])

    for model_name, preds in results.items():
        acc = sum(preds) / len(preds)
        print(f"{model_name} accuracy: {acc:.4f}")
    return results