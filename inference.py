# inference.py
"""
1. 사전과 정답지 간 비교
   정탐 : 사전라벨 <-> 정답지라벨 동일
   오탐 : 사전에 있으나 정답지에 없는경우 -> 사전에서 즉시 제외
   미탐 : 사전에 없으나 정답지에 있는경우

2. NER/regex기반 정보리스트와 정답지 간 비교
   정탐 : 정보리스트 <-> 정답지 동일 -> 사전 등재 후보 리스트로 올림
   오탐 : 정보리스트로 추출되었으나 정답지에 없는경우
   미탐 : 정보리스트로 추출되지 않으나 정답지에 있는 경우

3. Span 추출 알고리즘 기반으로 정답지 비교
   정탐 : 추출된 Span의 Inference 결과가 정답지와 동일한 경우
   오탐 : 추출된 Span의 Inference 결과가 정답지와 다른 경우
   미탐 : 추출된 Span의 Inference 결과가 "일반"인데, 정답지에 "일반"이 아닌 상태로 존재하는 경우
"""

def validation_1(dataloader, conn):
    pass
   

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