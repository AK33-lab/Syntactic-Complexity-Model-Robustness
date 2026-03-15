import random
import numpy as np
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=67):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
set_seed(67)

# Shared embedding backbone for MLP and RNN
EMBED_MODEL_NAME = "textattack/roberta-base-MNLI"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
embed_model.eval()

def get_embedding(premise, hypothesis):
    """Encode premise and hypothesis separately, return [p, h, |p-h|, p*h] as a (1, 4, 768) sequence."""
    inputs_p = tokenizer(premise, return_tensors="pt", truncation=True,
                         max_length=64, padding="max_length").to(DEVICE)
    inputs_h = tokenizer(hypothesis, return_tensors="pt", truncation=True,
                         max_length=64, padding="max_length").to(DEVICE)
    with torch.no_grad():
        p_emb = embed_model(**inputs_p).last_hidden_state[:, 0, :]  # (1, 768)
        h_emb = embed_model(**inputs_h).last_hidden_state[:, 0, :]  # (1, 768)
    diff = (p_emb - h_emb).abs()
    prod = p_emb * h_emb
    return torch.stack([p_emb, h_emb, diff, prod], dim=1)  # (1, 4, 768)

# MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768 * 4, hidden_dim=256, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # (batch, 4, 768) -> (batch, 3072)
        return self.network(x)

# RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=3):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x is (batch, 4, 768) — a real 4-step sequence
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])

# Pretrained Transformer Models
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

def load_roberta():
    """Encoder-only transformer fine-tuned on MultiNLI."""
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(DEVICE)
    model.eval()
    return tokenizer, model

def load_bart():
    """Encoder-decoder transformer fine-tuned on MultiNLI."""
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(DEVICE)
    model.eval()
    return tokenizer, model

# Training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class NLIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        emb = get_embedding(ex["premise"], ex["hypothesis"])
        return emb.squeeze(0), torch.tensor(ex["label"])
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model(model, data, save_path, epochs=3, batch_size=32):
    model = model.to(DEVICE)

    generator = torch.Generator()
    generator.manual_seed(67)

    dataset = NLIDataset(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=generator
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for embs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            embs = embs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(embs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved weights to {save_path}")
    return epoch_losses

#Inference
def predict_mlp(model, premise, hypothesis):
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        emb = get_embedding(premise, hypothesis)
        output = model(emb)
        return torch.argmax(output, dim=1).item()

def predict_rnn(model, premise, hypothesis):
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        emb = get_embedding(premise, hypothesis)
        output = model(emb)
        return torch.argmax(output, dim=1).item()

def predict_transformer(tokenizer, model, premise, hypothesis):
    model = model.to(DEVICE)
    model.eval()
    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()