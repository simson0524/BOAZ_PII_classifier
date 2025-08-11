# PIIClassifier/train.py

from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from model import SpanPIIClassifier
from dataset import SpanClassificationDataset, load_all_json
import torch
import json
import os

# Label mapping
label_2_id = {"일반" : 0, "개인" : 1}
id_2_label = {0 : "일반", 1 : "개인"}

# Train Loop
def train_loop(model, dataloader, optimizer, device, tqdm_disable=False):
    model.train()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, total=len(dataloader), desc="train", disable=tqdm_disable):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_start = batch["token_start"].to(device)
        token_end = batch["token_end"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_start=token_start,
                        token_end=token_end)
        
        loss = loss_fn(outputs["logits"], labels) # Logits : [0.87, 0.13] -> [1, 0]  /  GT : [1, 0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation
def evaluate(model, dataloader, device, tqdm_disable=False):
    model.eval()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc='evaluation', disable=tqdm_disable):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_start = batch["token_start"].to(device)
            token_end = batch["token_end"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_start=token_start,
                            token_end=token_end)
            
            loss = loss_fn(outputs["logits"], labels)
            total_loss += loss.item()

            pred_labels = torch.argmax(outputs['logits'], dim=-1)
            preds.extend(pred_labels.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return avg_loss, precision, recall, f1

# 여기부터는 수정요망
if __name__ == "__main__":
    # Load model and tokenizer
    model_name = "klue/roberta-base"
    model = AutoModel.from_pretrained( model_name )
    tokenizer = AutoTokenizer.from_pretrained( model_name, use_fast=True )
    print(f"=====[ MODEL CONFIG INFO ]=====\n{AutoConfig.from_pretrained( model_name )}\n\n")

    # Set train config
    tqdm_disable = False
    train_name = "250811_03"
    batch_size = 128
    num_epochs = 30
    learning_rate = 1e-5
    max_length = 256
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=====[ Train CONFIG INFO ]====\nBatch_size : {batch_size}\nNum_epochs : {num_epochs}\nLearning_rate : {learning_rate}\nMax_length : {max_length}\nDevice : {device}\n\n")

    # Load data
    json_data_dir = "../Data/samples_60"
    all_json_data = load_all_json(json_data_dir)

    # Split data into train and valid
    train_json, valid_json = {}, {}
    train_json["data"], valid_json["data"], train_json["annotations"], valid_json["annotations"] = train_test_split(all_json_data["data"],
                                                                                                                    all_json_data["annotations"],
                                                                                                                    test_size=0.2,
                                                                                                                    random_state=42)
    # Dataset
    train_dataset = SpanClassificationDataset(train_json, tokenizer, label_2_id, max_length)
    valid_dataset = SpanClassificationDataset(valid_json, tokenizer, label_2_id, max_length)

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = SpanPIIClassifier(model, 
                              num_labels=2).to( device )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate)
    
    # Train & Evaluation
    best_f1 = 0
    for epoch in range(1, num_epochs+1):
        print(f"\n==== Epoch {epoch} ====")
        train_loss = train_loop(model, train_loader, optimizer, device, tqdm_disable)
        val_loss, val_prec, val_rec, val_f1 = evaluate(model, valid_loader, device, tqdm_disable)

        print(f"[Train] Loss: {train_loss:.4f}")
        print(f"[Valid] Loss: {val_loss:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs("../Checkpoints", exist_ok=True)
            model_path = os.path.join("../Checkpoints", f"{train_name}_best_model_{epoch}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved! @[{model_path}]")