import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

MODEL_NAME = "klue/bert-base"
NUM_LABELS = 2
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
TRAIN_SUBSET = 3000
EVAL_SUBSET = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class APEACHDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data(tokenizer):
    ds = load_dataset("jason9693/APEACH")
    train = ds["train"]
    val = ds["test"]
    if TRAIN_SUBSET:
        train = train.shuffle(seed=42).select(range(min(TRAIN_SUBSET, len(train))))
    if EVAL_SUBSET:
        val = val.shuffle(seed=42).select(range(min(EVAL_SUBSET, len(val))))
    train_set = APEACHDataset(train["text"], train["class"], tokenizer, MAX_LEN)
    val_set = APEACHDataset(val["text"], val["class"], tokenizer, MAX_LEN)
    return train_set, val_set


def train(model, train_loader):
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    total_step = len(train_loader)
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}"
                )


def evaluate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total else 0
    print(f"Validation Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def save_model(model):
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)
    train_set, val_set = load_data(tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    train(model, train_loader)
    evaluate(model, val_loader)
    save_model(model)
