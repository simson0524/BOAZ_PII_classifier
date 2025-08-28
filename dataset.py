# PIIClassifier/dataset.py

from torch.utils.data import Dataset
from konlpy.tag import Okt
from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd
import random
import torch
import json
import os


class SpanClassificationTrainDataset(Dataset):
    def __init__(self, train_name, json_data, tokenizer, label_2_id, use_okt=False, max_length=256):
        self.samples = {data['id']: data for data in json_data["data"]}
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.max_length = max_length
        self.train_name = train_name
        self.instances = []
        self.okt = Okt()
        self._create_instances()

        """
        1. 일단 Okt() -> Span추출 알고리즘을 이용하여 span후보가 될 수 있는 것들을 모두 추출 -> 모두 "일반(0)"으로 일단 annotation
        2. 추출된 Span후보들 중 정답지와 비교하여 만족하는 것에 알맞은 클래스로 annotation
        """

    def _create_instances(self):
        span_truncated_sent = 0
        samples_num = [0 for _ in range( len(label_2_id) )]

        # self.samples에 접근하여 각 문장데이터별로 span후보 모두 추출
        for id, sample_data in tqdm(self.samples.items(), desc="Create Trainset instances"):
            sentence = sample_data['sentence']

            # 토큰화(tokenizing)
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offset_mapping=True,
                padding='max_length'
            )
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            # okt를 이용한 Span후보가 될 수 있는 POS tag 목록
            target_pos = {"Noun", "Number", "Email", "URL", "Foreign", "Alpha"}
            POS_in_sentence = self.okt.pos( sentence )

            # TODO target_pos에 있는 친구들이 공백 없이 붙어있으면 한 span으로 취급(여부 논의 필수, 현재 단순히 POS_in_sentence 단위로만 진행중)
            str_find_start_idx = 0
            for span_text, pos in POS_in_sentence:
                if pos in target_pos:
                    char_start = sentence.find(span_text, str_find_start_idx)
                    if char_start == -1:
                        continue
                    char_end = char_start + len(span_text)
                    str_find_start_idx = char_end

                    # char 단위 인덱스를 token 단위 인덱스로 변환
                    token_start, token_end = None, None
                    for i, (offset_char_start, offset_char_end) in enumerate(offset_mapping):
                        if offset_char_start <= char_start < offset_end_char:
                            token_start = i
                        if offset_char_start < char_end <= offset_end_char:
                            token_end = i+1
                    if (token_start is None) or (token_end is None): # truncation으로 인해 해당 Span이 잘린경우 사용불가하므로
                        continue
                    span_ids = input_ids[token_start:token_end]
                    decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                    # TODO : span_text가 정답지에 있는지 혜주가 구현한 디비 조회 함수로 찾기
                    if span_text in DB:
                        label_id = #DB의 정보 열의 value참조
                        samples_num[label_id] += 1
                    else:
                        label_id = label_to_id["일반"]
                        samples_num[label_id] += 1

                    self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                    })
        
        print(f"[Total instances]\nLabel & ids : {label_to_id}\nnums : {samples_num}\n\n")

        instance_df = pd.DataFrame(self.instances)

        instance_df.to_csv(f'../DatasetInstanceSamples/{self.train_name}_train_dataset_samples.csv', index=False)
        

    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "idx": torch.tensor(idx),
            "sentence": item['sentence'],
            "input_ids": torch.tensor(item["input_ids"]),
            "decoded_input_ids": item["decoded_input_ids"],
            "attention_mask": torch.tensor(item["attention_mask"]),
            "span_text": item["span_text"],
            "span_ids": torch.tensor(item['span_ids']),
            "decoded_span_ids": item["decoded_span_ids"],
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "label": torch.tensor(item["label"]),
        }
    
    def __len__(self):
        return len(self.instances)


class SpanClassificationTestDataset(Dataset):
    def __init__(self, test_name, json_data, tokenizer, label_2_id, use_okt=False, max_length=256):
        self.samples = {data['id']: data for data in json_data["data"]}
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.max_length = max_length
        self.test_name = test_name
        self.instances = []
        self.okt = Okt()
        self._create_instances()
 
        

    def _create_instances(self):
        # TODO : confusion matrix는 여기게 아니라 test코드로 보내주기
        # mat[0][1] -> axis0 : ground truth | axis1 : pred
        # 사전기반 axis0 : 정답지의 라벨 | axis1 : SPAN_TEXT가 포함된 각 사전의 라벨
        valid_1_confusion_matrix = [[0 for _ in range( len(label_2_id) )] for _ in range( len(label_2_id) )]
        # NER/REGEX기반 axis0 : 정답지의 라벨 | axis1 : 각 NER/REGEX가 포함된 라벨
        valid_2_confusion_matrix = [[0 for _ in range( len(label_2_id) )] for _ in range( len(label_2_id) )]
        # 일반적인 confusion matrix
        valid_3_confusion_matrix = [[0 for _ in range( len(label_2_id) )] for _ in range( len(label_2_id) )]


        # self.samples에 접근하여 검증1, 검증2, 검증3별로 인스턴스 생성하기
        for id, sample_data in tqdm(self.samples.items(), desc="Create Test instances"):
            sentence = sample_data['sentence']

            # 토큰화(tokenizing)
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offset_mapping=True,
                padding='max_length'
            )
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            # 각 검증단계 별 기 추출된 토큰은 이후 단계에서 추출되면 안되므로,
            special_ids = set(self.tokenizer.all_special_ids)
            is_extracted = [ id in special_ids for id in input_ids ]

            # 검증 1 | 사전을 참고하여 사전에 있는 단어만 일단 self.instances에 append
            # TODO : SPAN_TEXT를 개인정보/기밀정보/준식별자 DB에서 추출
            # for 디비 전체 순회(시간 꽤 걸리겠는데..)
            #       if 맞는부분이 있으면
            self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 1
                    })

            # 검증 2 | NER/REGEX 추출 알고리즘 갖고오기
            # TODO
            # [정규표현식]
            # /DataPreprocessLogics/regex_based_doc_parsing/pii_detector/main.py | run_pii_detection()
            # [NER]
            # 
            self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 2
                    })

            # 검증 3 | Span추출 알고리즘
            # TODO
            self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 3
                    })

            # okt를 이용한 Span후보가 될 수 있는 POS tag 목록
            target_pos = {"Noun", "Number", "Email", "URL", "Foreign", "Alpha"}
            POS_in_sentence = self.okt.pos( sentence )

            # TODO target_pos에 있는 친구들이 공백 없이 붙어있으면 한 span으로 취급(여부 논의 필수, 현재 단순히 POS_in_sentence 단위로만 진행중)
            str_find_start_idx = 0
            for span_text, pos in POS_in_sentence:
                if pos in target_pos:
                    char_start = sentence.find(span_text, str_find_start_idx)
                    if char_start == -1:
                        continue
                    char_end = char_start + len(span_text)
                    str_find_start_idx = char_end

                    # char 단위 인덱스를 token 단위 인덱스로 변환
                    token_start, token_end = None, None
                    for i, (offset_char_start, offset_char_end) in enumerate(offset_mapping):
                        if offset_char_start <= char_start < offset_end_char:
                            token_start = i
                        if offset_char_start < char_end <= offset_end_char:
                            token_end = i+1
                    if (token_start is None) or (token_end is None): # truncation으로 인해 해당 Span이 잘린경우 사용불가하므로
                        continue
                    span_ids = input_ids[token_start:token_end]
                    decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                    # TODO : span_text가 정답지에 있는지 혜주가 구현한 디비 조회 함수로 찾기
                    if span_text in DB:
                        label_id = #DB의 정보 열의 value참조
                        samples_num[label_id] += 1
                    else:
                        label_id = label_to_id["일반"]
                        samples_num[label_id] += 1

                    self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                    })
        
        print(f"[Total instances]\nLabel & ids : {label_to_id}\nnums : {samples_num}\n\n")

        instance_df = pd.DataFrame(self.instances)

        instance_df.to_csv(f'../DatasetInstanceSamples/{self.train_name}_train_dataset_samples.csv', index=False)
        

    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "idx": torch.tensor(idx),
            "sentence": item['sentence'],
            "input_ids": torch.tensor(item["input_ids"]),
            "decoded_input_ids": item["decoded_input_ids"],
            "attention_mask": torch.tensor(item["attention_mask"]),
            "span_text": item["span_text"],
            "span_ids": torch.tensor(item['span_ids']),
            "decoded_span_ids": item["decoded_span_ids"],
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "label": torch.tensor(item["label"]),
            "validation_priority": item["validation_priority"]
        }
    
    def __len__(self):
        return len(self.instances)


class OriginSpanClassificationDataset(Dataset):
    def __init__(self, train_name, json_data, tokenizer, label_2_id, max_length=256):
        self.samples = {data['id']: data for data in json_data["data"]}
        self.annotations = {ann['id']: ann['annotations'] for ann in json_data['annotations']}
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.max_length = max_length
        self.train_name = train_name
        self.instances = []
        self.okt = Okt()
        self._create_instances()


    def _create_instances(self):
        span_truncated_sent = 0
        positive_samples = 0
        negative_samples = 0
        for sent_id, anns in self.annotations.items():
            sent = self.samples[sent_id]["sentence"]
            encoding = self.tokenizer(
                sent,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            used_token_spans = set()

            # 향후 일반 라벨이랑 밸런스를 맞춰주기 위함(random_selected_normal_instances에서 사용)
            num_anns = len(anns)

            # 1. annotation 라벨 처리 (char index → token index 변환이 필요[협의를 통해 결정완료])
            for ann in anns:
                start_char, end_char, label, score = ann['start'], ann['end'], ann['label'], ann['score']
                score = float(score)

                if label not in self.label_2_id:
                    print(f"[{label}] label is not available")
                    continue

                # char index → token index 변환
                token_start, token_end = None, None
                for i, (start, end) in enumerate(offset_mapping):
                    if start <= start_char < end:
                        token_start = i
                    if start < end_char <= end:
                        token_end = i + 1
                if token_start is None or token_end is None:
                    span_truncated_sent += 1
                    # print(f"[NOTICE] Current Sentence is skipped due to span truncation\n{sent}\n\n")
                    continue

                label_id = self.label_2_id[label]
                used_token_spans.add((token_start, token_end))

                self.instances.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_start": token_start,
                    "token_end": token_end,
                    "label": label_id,
                    "sentence": sent,
                    "span_text": self.tokenizer.convert_tokens_to_string(
                        self.tokenizer.convert_ids_to_tokens(input_ids[token_start:token_end])
                    ),
                    "span_conf_score": score
                })

                positive_samples += 1

            # 2. 명사 → 일반 라벨링 (char index 기반 → token index 변환 필요)
            normal_instances = []

            nouns = self.okt.nouns(sent)
            for noun in nouns:
                if noun.strip() == "":
                    continue

                # 동일 명사 반복 시 중복 제거용
                for start_idx in [i for i in range(len(sent)) if sent.startswith(noun, i)]:
                    end_idx = start_idx + len(noun)
                    if end_idx > len(sent):
                        continue

                    token_start, token_end = None, None
                    for i, (s, e) in enumerate(offset_mapping):
                        if s == e:
                            continue # [CLS], [SEP], [PAD]은 offset_mapping -> (0,0)
                        if s <= start_idx < e:
                            token_start = i
                        if s < end_idx <= e:
                            token_end = i + 1
                    if token_start is None or token_end is None:
                        continue

                    if (token_start, token_end) in used_token_spans:
                        continue

                    label_id = self.label_2_id.get("일반", None)
                    if label_id is None:
                        continue 

                    span_text = self.tokenizer.convert_tokens_to_string(
                            self.tokenizer.convert_ids_to_tokens(input_ids[token_start:token_end])
                        )

                    if len(span_text) == 1:
                        continue

                    normal_instances.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "sentence": sent,
                        "span_text": span_text,
                        "span_conf_score": 1
                    })
                    used_token_spans.add((token_start, token_end))

            # label balance를 위한 조치
            num_random_samples = min(num_anns, len(normal_instances))
            random_selected_normal_instances = random.sample(normal_instances, num_random_samples)
            self.instances.extend( random_selected_normal_instances )

            negative_samples += len( random_selected_normal_instances )
        
        print(f"[Total instances]\nPositive : {positive_samples}\nNegative : {negative_samples}")

        instance_df = pd.DataFrame(self.instances)

        instance_df.to_csv(f'../DatasetInstanceSamples/{self.train_name}_dataset_samples.csv', index=False)
        

    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "idx": torch.tensor(idx),
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "labels": torch.tensor(item["label"]),
            "span_conf_score": torch.tensor(item["span_conf_score"])
        }
    
    def __len__(self):
        return len(self.instances)


def load_all_json(json_dir="../Data"):
    """Dataset 클래스에 로딩하기 위해 모든 문장별 JSON 파일의 데이터를 병합하는 함수

    Args:
        json_dir (str, optional): 문장별 JSON 파일이 있는 폴더 경로. Defaults to "../Data".

    Returns:
        dict: 모든 데이터들이 병합된 결과물
    """
    all_data = {"data": [],
                "annotations": []}
    
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(json_dir, file_name), "r", encoding='utf-8') as f:
                json_file = json.load(f)
                all_data["data"].append( json_file["data"] )
                all_data["annotations"].extend( json_file["annotations"] )
    
    return all_data