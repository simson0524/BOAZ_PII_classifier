# train.py

### TODO : 데이터셋(JSON)에 알맞게 dataloader 설계
### TODO : Distributed Data Parallel을 사용한 병렬학습 구현

from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from dataset import PIISpanDataset, collate_fn
from model import SpanPIIClassifier
import torch.nn.functional as F
import torch
import json
import os

# Train Loop
def train_loop(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, total=len(dataloader), desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        span_indices = batch["span_indices"]
        span_labels = batch["span_labels"]

        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       span_indices_batch=span_indices)
        
        loss = 0

        for pred, label in zip(logits, span_labels):
            label = label.to(device)
            loss += F.binary_cross_entropy_with_logits(pred, label)
        
        loss /= len(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            span_indices = batch['span_indices']
            span_labels = batch['span_labels']

            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           span_indices_batch=span_indices)
            
            for pred, label in zip(logits, span_labels):
                label = label.to(device)
                loss = F.binary_cross_entropy_with_logits(pred, label)
                total_loss += loss.item()

                probs = torch.sigmoid(pred)
                total_preds.extend((probs > 0.5).int().cpu().tolist())
                total_labels.extend(label.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    precision = precision_score(total_labels, total_preds, zero_division=0)
    recall = recall_score(total_labels, total_preds, zero_division=0)
    f1 = f1_score(total_labels, total_preds, zero_division=0)

    return avg_loss, precision, recall, f1

if __name__ == "__main__":
    # Load SKT/KoBERT model and tokenizer
    model_name = "skt/kobert-base-v1"
    model = AutoModel.from_pretrained( model_name )
    tokenizer = AutoTokenizer.from_pretrained( model_name )

    # Set train config
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data & dataloader
    # TODO #
    train_loader = "TODO"
    valid_loader = "TODO"

    # Model
    model = SpanPIIClassifier(model).to( device )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate)
    
    # Train & Evaluation
    best_f1 = 0
    for epoch in range(1, num_epochs+1):
        print(f"\n==== Epoch {epoch} ====")
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, val_prec, val_rec, val_f1 = evaluate(model, valid_loader, device)

        print(f"[Train] Loss: {train_loss:.4f}")
        print(f"[Valid] Loss: {val_loss:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("best_models", exist_ok=True)
            model_path = os.path.join("best_models", f"000000_000_best_model_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved! @[{model_path}]")