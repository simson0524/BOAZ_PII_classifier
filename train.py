# PIIClassifier/train.py

from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from PIIClassifier.model import SpanPIIClassifier
from PIIClassifier.train_dataset import SpanClassificationTrainDataset, load_all_json
from DataPreprocessLogics.DBMS.create_dbs import *
from DataPreprocessLogics.DBMS.edit_dbs import *
from datetime import datetime
import pandas as pd
import torch
import yaml
import json
import os


# # 커스텀 CELoss
# def soft_cross_entropy_with_confidence(logits, target_indices, conf_scores):
#     num_classes = logits.size(1)
    
#     # 커스텀 원 핫 라벨 * conf_scores
#     target_one_hot = torch.nn.functional.one_hot(target_indices, num_classes=num_classes).float()
#     target_soft = target_one_hot * conf_scores.unsqueeze(1)

#     # CE 계산
#     log_probs = torch.nn.functional.log_softmax(logits, dim=1)
#     loss = -(target_soft * log_probs).sum(dim=1)

#     return loss.mean()

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
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_start=token_start,
                        token_end=token_end)
        
        loss = loss_fn(outputs["logits"], labels) # Logits : [0.87, 0.13] -> [1, 0]  /  GT : [0, 1]
        # loss = soft_cross_entropy_with_confidence(outputs["logits"], labels, scores)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

# Evaluation
def evaluate(conn, experiment_name, model, epoch, dataloader, device, label_2_id, id_2_label, tqdm_disable=False, is_best_model=False, tokenizer=None, train_name='test'):
    model.eval()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    preds, targets = [], []

    # 불일치 샘플 목록
    model_train_sent_dataset_log = []

    with torch.no_grad():        
        for batch in tqdm(dataloader, total=len(dataloader), desc='evaluation', disable=tqdm_disable):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_start = batch["token_start"].to(device)
            token_end = batch["token_end"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_start=token_start,
                            token_end=token_end)
            
            loss = loss_fn(outputs["logits"], labels)
            total_loss += loss.item()

            pred_labels = torch.argmax(outputs['logits'], dim=-1)
            preds.extend(pred_labels.cpu().tolist())
            targets.extend(labels.cpu().tolist())

            batch_size = labels.size(0)
            for i in range(batch_size):
                # model_train_sent_dataset_log 테이블에 들어가는 scheme
                model_train_sent_dataset_log_scheme = (
                    experiment_name,
                    batch['sentence_id'][i],
                    epoch,
                    batch['sentence'][i],
                    batch['span_text'][i],
                    batch['idx'][i],
                    id_2_label(labels[i]),
                    id_2_label(pred_labels[i]),
                    batch['file_name'][i],
                    batch['sentence_seq'][i]
                )
                model_train_sent_dataset_log.append( model_train_sent_dataset_log_scheme )

    # DB(model_train_sent_dataset_log)에 정보 추가하기
    insert_many_rows(conn, "model_train_sent_dataset_log", model_train_sent_dataset_log)

    metric = [ [0 for i in range(len(label_2_id))] for _ in range(len(label_2_id)) ]

    for i, (pred, gt) in enumerate(zip(preds, targets)):
        metric[pred][gt] += 1
                            
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    return avg_loss, precision, recall, f1, metric


def train(experiment_name, config):
    # 모델학습 시작시간
    start_time = datetime.now()

    # DB connection
    conn = get_connection()

    # SETTING
    model_name = config['model']['model_name']
    model = AutoModel.from_pretrained( model_name )
    tokenizer = AutoTokenizer.from_pretrained( model_name, use_fast=True )
    print(f"=====[ MODEL CONFIG INFO ]=====\n{AutoConfig.from_pretrained( model_name )}\n\n")

    # Train config
    tqdm_disable = False
    train_name = experiment_name
    batch_size = config['exp']['batch_size']
    num_epochs = config['exp']['num_epochs']
    learning_rate = float(config['exp']['learning_rate'])
    is_pii = config['exp']['is_pii']
    max_length = 256
    device = torch.device( config['exp']['device'] )
    print(f"=====[ Train CONFIG INFO ]====\nBatch_size : {batch_size}({type(batch_size)})\nNum_epochs : {num_epochs}({type(num_epochs)})\nLearning_rate : {learning_rate}({type(learning_rate)})\nMax_length : {max_length}({type(max_length)})\nDevice : {device}({type(device)})\n\n")

    # Load data
    train_data_dir = config['data']['train_data_dir']
    validation_data_dir = config['data']['test_data_dir']
    train_json = load_all_json( train_data_dir )
    valid_json = load_all_json( validation_data_dir )

    # Dataset 
    if is_pii:
        label_2_id = config['label_mapping']['pii_label_2_id']
        id_2_label = config['label_mapping']['pii_id_2_label']
    else:
        label_2_id = config['label_mapping']['confid_label_2_id']
        id_2_label = config['label_mapping']['confid_id_2_label']
    train_dataset = SpanClassificationTrainDataset(
        train_name=train_name,
        json_data=train_json, 
        tokenizer=tokenizer, 
        label_2_id=label_2_id,
        sampling_ratio=1.0, 
        is_valid=True, 
        is_pii=is_pii, 
        max_length=max_length
        )
    valid_dataset = SpanClassificationTrainDataset(
        train_name=train_name,
        json_data=valid_json, 
        tokenizer=tokenizer, 
        label_2_id=label_2_id,
        sampling_ratio=1.0, 
        is_valid=False, 
        is_pii=is_pii, 
        max_length=max_length
        )
    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model
    if is_pii:
        model = SpanPIIClassifier(model, 
                                  num_labels=3).to( device )
    else:
        model = SpanPIIClassifier(model, 
                                  num_labels=2).to( device )
        

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate)
    
    # Train & Evaluation
    best_f1 = 0
    best_prec = 0
    pest_rec = 0
    best_metric = ''
    best_epoch = 0
    for epoch in range(1, num_epochs+1):
        print(f"\n==== Epoch {epoch} ====")
        train_loss = train_loop(model, train_loader, optimizer, device, tqdm_disable)
        val_loss, val_prec, val_rec, val_f1, val_metric = evaluate(conn, experiment_name, model, epoch, valid_loader, device, label_2_id, id_2_label, tqdm_disable)

        print(f"[Train] Loss: {train_loss:.4f}")
        print(f"[Valid] Loss: {val_loss:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f} | Metric: {val_metric}")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_prec = val_prec
            best_rec = val_rec
            best_metric = val_metric
            best_epoch = epoch
            os.makedirs("Checkpoints", exist_ok=True)
            model_path = os.path.join("Checkpoints", f"{train_name}_best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved! @[{model_path}]")

    # 모델학습 종료시간
    end_time = datetime.now()

    # 모델학습 소요시간
    duration = end_time - start_time

    # model_train_performance 테이블에 들어가는 scheme
    model_train_performance_log = [(
        experiment_name,
        start_time,
        end_time,
        model_path,
        best_epoch,
        best_prec,
        best_rec,
        best_f1,
        best_metric
    )]

    # DB(model_train_performance)에 정보 추가하기
    insert_many_rows(conn, "model_train_performance", model_train_performance_log)

    conn.close()

    return duration