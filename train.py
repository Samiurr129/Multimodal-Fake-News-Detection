"""Multimodal fake news detection: BERT for text + a small CNN for images.

Expects a CSV with columns: text, image_path, label (0 = real, 1 = fake).
image_path may be absolute or relative to the CSV's folder.

Train and evaluate:
    python train.py --data data/fakenews_dataset.csv --epochs 5

Quick check that everything runs (tiny model, synthetic data):
    python make_demo_data.py
    python train.py --data demo_data/demo.csv --bert-model google/bert_uncased_L-2_H-128_A-2 --epochs 1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

IMAGE_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class FakeNewsDataset(Dataset):
    def __init__(self, frame, tokenizer, image_root, max_length=256):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_root = Path(image_root)
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(str(row["text"]), padding="max_length", max_length=self.max_length,
                             truncation=True, return_tensors="pt")
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.image_root / path
        image = TRANSFORM(Image.open(path).convert("RGB"))
        return (enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), image,
                torch.tensor(int(row["label"]), dtype=torch.long))


class MultimodalFakeNewsClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 256), nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, 2)   # real / fake

    def forward(self, input_ids, attention_mask, images):
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        fused = torch.cat((torch.relu(self.text_fc(cls)), self.cnn(images)), dim=1)
        return self.fc(self.dropout(fused))


def run_epoch(model, loader, device, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, labels_all, preds_all = 0.0, [], []
    with torch.set_grad_enabled(training):
        for input_ids, mask, images, labels in loader:
            input_ids, mask, images, labels = (t.to(device) for t in (input_ids, mask, images, labels))
            logits = model(input_ids, mask, images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(logits.argmax(dim=1).cpu().tolist())
    y, p = np.array(labels_all), np.array(preds_all)
    return total_loss / len(y), y, p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bert-model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--save", type=Path, default=Path("model.pt"))
    args = ap.parse_args()

    torch.manual_seed(42)
    df = pd.read_csv(args.data)
    missing = {"text", "image_path", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.data} is missing columns: {sorted(missing)}")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    root = args.data.parent
    train_loader = DataLoader(FakeNewsDataset(train_df, tokenizer, root, args.max_length),
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(FakeNewsDataset(test_df, tokenizer, root, args.max_length),
                             batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFakeNewsClassifier(args.bert_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, y, p = run_epoch(model, train_loader, device, criterion, optimizer)
        print(f"Epoch {epoch}/{args.epochs}  train loss {loss:.4f}  train acc {(y == p).mean():.2%}")

    loss, y, p = run_epoch(model, test_loader, device, criterion)
    print(f"\nTest loss {loss:.4f}")
    print(f"Precision {precision_score(y, p, zero_division=0):.3f}  "
          f"Recall {recall_score(y, p, zero_division=0):.3f}  F1 {f1_score(y, p, zero_division=0):.3f}")
    print(classification_report(y, p, target_names=["real", "fake"], zero_division=0))
    torch.save(model.state_dict(), args.save)
    print(f"Saved model weights to {args.save}")


if __name__ == "__main__":
    main()
