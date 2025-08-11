# inference.py
"""
수정중
"""

# Test
def test(model, dataloader, device, tqdm_disable=False):
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