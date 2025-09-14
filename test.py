# PIIClassifier/test.py

from DataPreprocessLogics.DBMS.db_sdk import get_connection, fetch_rows, delete_row, create_metric_tables, add_metric_rows, create_prediction_tables, truncate_tables, add_prediction_rows
from PIIClassifier.test_dataset import SpanClassificationTestDataset, load_all_json
from PIIClassifier.model import SpanPIIClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import torch
import yaml

"""
[검증1] 사전과 정답지 간 비교
    정탐 : 사전라벨 <-> 정답지라벨 동일
    오탐 : 사전에 있으나 정답지에 없는경우 -> 사전에서 즉시 제외
    미탐 : 사전에 없으나 정답지에 있는경우

[검증2] NER/regex기반 정보리스트와 정답지 간 비교
    정탐 : 정보리스트 <-> 정답지 동일 -> 사전 등재 후보 리스트로 올림
    오탐 : 정보리스트로 추출되었으나 정답지에 없는경우
    미탐 : 정보리스트로 추출되지 않으나 정답지에 있는 경우

[검증3] Span 추출 알고리즘 기반으로 정답지 비교
    정탐 : 추출된 Span의 Inference 결과가 정답지와 동일한 경우
    오탐 : 추출된 Span의 Inference 결과가 정답지와 다른 경우
    미탐 : 추출된 Span의 Inference 결과가 "일반"인데, 정답지에 "일반"이 아닌 상태로 존재하는 경우
"""

def test_1(dataloader, conn, label_2_id):
    hit, wrong, mismatch = [], [], []

    # column_name = [정탐, 오탐, 미탐]
    # row_name = label_id of values
    metric = [ [0, 0, 0] for _ in range(len(label_2_id)) ]
    
    for label, _ in label_2_id.items():
        # "개인정보", "기밀정보"사전만 대조하므로!
        if label == "개인정보" or label == "기밀정보":
            for batch in tqdm(dataloader, desc=f"[검증1] {label}사전 검증중..."):
                batch_size = len( batch['sentence'] )
                for i in range( batch_size ):
                    curr_span_text = batch['span_text'][i]
                    curr_label_id = batch['label'][i].item()
                    curr_dataset_idx = batch['idx'][i].item()
                    
                    # 현재 사전과 데이터의 라벨이 같은경우에만 아래 로직 실행해야 함
                    if (label_2_id[label] == curr_label_id) and (batch['validation_priority'][i] == 1):
                        # 현재 사전에 SPAN_TEXT가 있는지 조회
                        dict_result = fetch_rows(
                            conn=conn,
                            table_name=label,
                            column_name="단어",
                            keyword=curr_span_text
                            ) # return : [(단어, 부서명, 문서명, 단어유형, 구분), ...]
                        master_result = fetch_rows(
                            conn=conn,
                            table_name="정답지",
                            column_name="단어",
                            keyword=curr_span_text
                            ) # return : [(단어, 부서명, 문서명, 단어유형, 구분), ...]

                        # 사전 and 정답지에 있는 경우(정탐)
                        if dict_result and master_result:
                            flag = False
                            for _, _, _, _, abs_label in master_result:
                                if abs_label == label:
                                    flag = True
                                    break
                            if flag:
                                hit.append((curr_span_text, curr_dataset_idx))
                                metric[label_2_id[label]][0] += 1
                                continue

                        # 사전에만 있는 경우(오탐, 사전에서 제외, validation_priority 2로 변경)
                        if dict_result and not master_result:
                            delete_row(
                                conn=conn,
                                table_name=label,
                                word=curr_span_text
                            )
                            wrong.append((curr_span_text, curr_dataset_idx))
                            metric[label_2_id[label]][1] += 1
                            continue

                        # 정답지에만 있는 경우(미탐, validation_priority 2로 변경)
                        if master_result and not dict_result:
                            mismatch.append((curr_span_text, curr_dataset_idx))
                            metric[label_2_id[label]][2] += 1
                            continue
                    
    
    return metric, hit, wrong, mismatch


# TODO : 여기 좀 이상한가...? 지금?? 가져올 NER 클래스 목록 + REGEX이 없네
def test_2(dataloader, conn, label_2_id):
    hit, wrong, mismatch = [], [], []

    # column_name = [정탐, 오탐, 미탐]
    # row_name = label_id of values
    metric = [ [0, 0, 0] for _ in range(len(label_2_id)) ]

    for label, _ in label_2_id.items():
        # "일반정보"인 NER/REGEX는 없으므로!
        if label == "일반정보":
            continue
        for batch in tqdm(dataloader, desc=f"[검증2] {label} 데이터 NER/REGEX 검증중..."):
            batch_size = len( batch['sentence'] )
            for i in range( batch_size ):
                curr_span_text = batch['span_text'][i]
                curr_label_id = batch['label'][i].item()
                curr_dataset_idx = batch['idx'][i].item()
                
                # 현재 사전과 데이터의 라벨이 같은경우에만 아래 로직 실행해야 함
                if (label_2_id[label] == curr_label_id) and (batch['validation_priority'][i] == 2):
                    # 현재 사전에 SPAN_TEXT가 있는지 조회
                    dict_result = fetch_rows(
                        conn=conn,
                        table_name=label,
                        column_name="단어",
                        keyword=curr_span_text
                        ) # return : [(단어, 부서명, 문서명, 단어유형, 구분), ...]
                    master_result = fetch_rows(
                        conn=conn,
                        table_name="정답지",
                        column_name="단어",
                        keyword=curr_span_text
                        ) # return : [(단어, 부서명, 문서명, 단어유형, 구분), ...]

                    # 사전 and 정답지에 있는 경우(정탐, 사전등재리스트)
                    if dict_result and master_result:
                        flag = False
                        for _, _, _, _, abs_label in master_result:
                            if abs_label == label:
                                flag = True
                                break
                        if flag:
                            hit.append((curr_span_text, curr_dataset_idx))
                            metric[label_2_id[label]][0] += 1
                            continue

                    # 사전에만 있는 경우(오탐, validation_priority 3으로 변경)
                    if dict_result and not master_result:
                        delete_row(
                            conn=conn,
                            table_name=label,
                            word=curr_span_text
                        )
                        wrong.append((curr_span_text, curr_dataset_idx))
                        metric[label_2_id[label]][1] += 1
                        continue

                    # 정답지에만 있는 경우(미탐, validation_priority 3으로 변경)
                    if master_result and not dict_result:
                        mismatch.append((curr_span_text, curr_dataset_idx))
                        metric[label_2_id[label]][2] += 1
                        continue
                
    # 여기서 hit의 요소들은 사전등재후보리스트
    return metric, hit, wrong, mismatch        


def test_3(model, device, dataloader, conn, label_2_id):
    model.eval()

    # 배치 별 샘플수가 다를 수 있으므로 sum후 마지막에 나눔
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # column_name = pred
    # row_name = GT
    metric = [ [0 for _ in range(len(label_2_id))] for _ in range(len(label_2_id)) ]

    hit, wrong, mismatch = [], [], []

    total_loss = 0
    total_cnt = 0

    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc="[검증3] Span 추출 알고리즘으로 추출한 데이터 검증중.."):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_start = batch["token_start"].to(device)
            token_end = batch["token_end"].to(device)
            labels = batch["label"].to(device)

            valid_priority_list = batch["validation_priority"]
            vp3_list = [ i for i, v in enumerate(valid_priority_list) if v == 3 ]

            # vp3_list가 없으면 "validation_priority"가 3인 친구가 없다는 의미이므로 건너띔
            if len( vp3_list ) == 0:
                continue

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_start=token_start,
                            token_end=token_end)
            logits = outputs["logits"]

            # validation priority==3 인 친구들만 추출
            vp3 = torch.tensor(vp3_list, device=device, dtype=torch.long)
            vp3_logits = torch.index_select(logits, dim=0, index=vp3)
            vp3_labels = torch.index_select(labels, dim=0, index=vp3)
            vp3_preds = torch.argmax(vp3_logits, dim=-1)

            # validation priority==3 인 친구들만 처리
            for i, vp3_idx in enumerate(vp3):
                gt = int( vp3_labels[i].item() )
                pred = int( vp3_preds[i].item() )
                metric[gt][pred] += 1

                # 검증 마지막 단계이므로 정탐/오탐/미탐 별 (SPAN_TEXT, GT_LABEL, PRED_LABEL)을 hit, wrong, mismatch에 append
                if gt == pred:
                    hit.append( (batch['span_text'][vp3_idx], gt, pred) )
                if pred != 0 and pred != gt:
                    wrong.append( (batch['span_text'][vp3_idx], gt, pred) )
                if pred == 0 and pred != gt:
                    mismatch.append( (batch['span_text'][vp3_idx], gt, pred) )

            # Loss
            loss = loss_fn(vp3_logits, vp3_labels)
            total_loss += loss.item()
            total_cnt  += vp3_labels.numel()

            # 메트릭 수집
            preds.extend(vp3_preds.detach().cpu().tolist())
            targets.extend(vp3_labels.detach().cpu().tolist())

    # 메트릭 계산
    avg_loss = total_loss / total_cnt
    precision = precision_score(targets, preds, average="macro", zero_division=0)
    recall = recall_score(targets, preds, average="macro", zero_division=0)
    f1 = f1_score(targets, preds, average="macro", zero_division=0)

    # 여기서 hit의 요소들은 사전등재후보리스트
    return avg_loss, precision, recall, f1, metric, hit, wrong, mismatch       


def test(config_file_path='run_config.yaml'):
    # DB connection
    conn = get_connection()

    # Load config from "test_config.yaml"
    with open(config_file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # PRE SETTING
    model_name = config['model']['model_name']
    model = AutoModel.from_pretrained( model_name )
    tokenizer = AutoTokenizer.from_pretrained( model_name, use_fast=True )

    # Test Config
    test_name = config['exp']['name']
    batch_size = config['exp']['batch_size']
    is_pii = config['exp']['is_pii']
    device = torch.device( config['exp']['device'] )
    max_length = 256
    print(f"=====[ Test CONFIG INFO ]====\nBatch_size : {batch_size}\nMax_length : {max_length}\nDevice : {device}\n\n")

    # Load model state
    if is_pii:
        state_path = config['model']['pii_state_dir']
        classifier = SpanPIIClassifier(
            pretrained_bert=model,
            num_labels=3
        ).to(device)
        state_dict = torch.load(state_path, map_location="cpu")
        label_2_id = config['label_mapping']['pii_label_2_id']
    else:
        state_path = config['model']['confid_state_dir']
        classifier = SpanPIIClassifier(
            pretrained_bert=model,
            num_labels=2
        ).to(device)
        state_dict = torch.load(state_path, map_location="cpu")
        label_2_id = config['label_mapping']['confid_label_2_id']
    classifier.load_state_dict( state_dict )

    # Dataset
    test_dataset_dir = config['data']['test_data_dir']
    all_json_test_data = load_all_json( test_dataset_dir )
    test_dataset = SpanClassificationTestDataset(
        test_name=test_name,
        json_data=all_json_test_data,
        tokenizer=tokenizer,
        label_2_id=label_2_id,
        is_pii=True,
        max_length=max_length
    )

    # Dataloader(검증1)
    test_dataloader_1 = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 검증1
    metric_1, hit_1, wrong_1, mismatch_1 = test_1(
        dataloader=test_dataloader_1,
        conn=conn,
        label_2_id=label_2_id
    )

    # 검증1의 오탐, 미탐 항목들의 validation_priority를 2로 만듦
    for _, idx in wrong_1:
        test_dataset.edit_validation_priority(
            idx=idx,
            edit_to=2
        )
    for _, idx in mismatch_1:
        test_dataset.edit_validation_priority(
            idx=idx,
            edit_to=2
        )
    
    # Dataloader(검증2)
    test_dataloader_2 = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 검증2
    metric_2, hit_2, wrong_2, mismatch_2 = test_2(
        dataloader=test_dataloader_2,
        conn=conn,
        label_2_id=label_2_id
    )

    # 검증2의 오탐, 미탐 항목들의 validation_priority를 3로 만듦
    for _, idx in wrong_2:
        test_dataset.edit_validation_priority(
            idx=idx,
            edit_to=3
        )
    for _, idx in mismatch_2:
        test_dataset.edit_validation_priority(
            idx=idx,
            edit_to=3
        )

    # Dataloader(검증3)
    test_dataloader_3 = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 검증3
    avg_loss, precision, recall, f1, metric_3, hit_3, wrong_3, mismatch_3 = test_3(
        model=model,
        device=device, 
        dataloader=test_dataloader_3, 
        conn=conn, 
        label_2_id=label_2_id
        )

    # TODO : hit_2, hit_3을 사전등재리스트에 올리고 일정 기준에 의해 사전에 추가
    # 사용할 NER 클래스명들 -> 무엇? 완철's DataPreprocessLogics/ner_based_doc_parsing/ner_main.py run_ner_detection()
    # 사용할 정규표현식들 -> 무엇? 혜주's DataPreprocessLogics/regex_based_doc_parsing/pii_detector/main.py run_regex_detector()

    # DB에 데이터 저장 할 테이블이 없는 경우를 대비하여 생성해주기
    # "모델_개인", "모델_기밀", "검증1_개인", "검증2_개인", "검증3_개인", "검증1_기밀", "검증2_기밀", "검증3_기밀"
    create_metric_tables(conn)
    timestamp = datetime.now()

    if is_pii:
        # 검증1(개인) 추가
        metric_1_row = [(test_name, timestamp, metric_1[0][0], metric_1[0][1], metric_1[0][2], metric_1[1][0], metric_1[1][1], metric_1[1][2], metric_1[2][0], metric_1[2][1], metric_1[2][2])]
        add_metric_rows(
            conn=conn,
            table_name="검증1_개인",
            rows=metric_1_row
        )
        
        # 검증2(개인) 추가
        metric_2_row = [(test_name, timestamp, metric_2[0][0], metric_2[0][1], metric_2[0][2], metric_2[1][0], metric_2[1][1], metric_2[1][2], metric_2[2][0], metric_2[2][1], metric_2[2][2])]
        add_metric_rows(
            conn=conn,
            table_name="검증2_개인",
            rows=metric_2_row
        )
        
        # 검증3(개인) 추가
        metric_3_row = [(test_name, timestamp, metric_3[0][0], metric_3[0][1], metric_3[0][2], metric_3[1][0], metric_3[1][1], metric_3[1][2], metric_3[2][0], metric_3[2][1], metric_3[2][2])]
        add_metric_rows(
            conn=conn,
            table_name="검증3_개인",
            rows=metric_3_row
        )
    else:
        # 검증1(기밀) 추가
        metric_1_row = [(test_name, timestamp, metric_1[0][0], metric_1[0][1], metric_1[0][2], metric_1[1][0], metric_1[1][1], metric_1[1][2], None, None, None)]
        add_metric_rows(
            conn=conn,
            table_name="검증1_기밀",
            rows=metric_1_row
        )
        
        # 검증2(기밀) 추가
        metric_2_row = [(test_name, timestamp, metric_2[0][0], metric_2[0][1], metric_2[0][2], metric_2[1][0], metric_2[1][1], metric_2[1][2], None, None, None)]
        add_metric_rows(
            conn=conn,
            table_name="검증2_기밀",
            rows=metric_2_row
        )
        
        # 검증3(기밀) 추가
        metric_3_row = [(test_name, timestamp, metric_3[0][0], metric_3[0][1], metric_3[0][2], metric_3[1][0], metric_3[1][1], metric_3[1][2], None, None, None)]
        add_metric_rows(
            conn=conn,
            table_name="검증3_기밀",
            rows=metric_3_row
        )

    # DB에 test 결과들을 저장할 테이블이 없는 경우를 대비해 생성하고 이전에 사용한 경우를 대비해 초기화해주기
    # "prediction"
    create_prediction_tables(conn)
    truncate_tables(conn, "prediction")

    # 검증3 오탐사항 "prediction"테이블에 추가
    for span_text, gt, pred in wrong_3:
        curr_row = [(test_name, timestamp, span_text, None, gt, pred)]
        add_prediction_rows(
            conn=conn,
            table_name='prediction',
            rows=curr_row
            )
    
    # 검증3 미탐사항 "prediction"테이블에 추가
    for span_text, gt, pred in mismatch_3:
        curr_row = [(test_name, timestamp, span_text, None, gt, pred)]
        add_prediction_rows(
            conn=conn,
            table_name='prediction',
            rows=curr_row
            )


    conn.close()