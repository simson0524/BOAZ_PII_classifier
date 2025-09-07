# PIIClassifier/test_dataset.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import Dataset
from konlpy.tag import Okt
from collections import defaultdict
from tqdm.auto import tqdm
from DataPreprocessLogics.DBMS.db_sdk import get_connection, find_words_in_sentence_for_doc
from DataPreprocessLogics.regex_based_doc_parsing.pii_detector.main import run_pii_detection
from DataPreprocessLogics.ner_based_doc_parsing.ner_module import run_ner
import pandas as pd
import random
import torch
import json
import os


class SpanClassificationTestDataset(Dataset):
    def __init__(self, test_name, json_data, tokenizer, label_2_id, is_pii=True, max_length=256):
        self.samples = {data['id']: data for data in json_data["data"]}
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.max_length = max_length
        self.test_name = test_name
        self.instances = []
        self.okt = Okt()
        self.is_pii = is_pii
        self._create_instances()
 
    """
    1. 사전에 있는 친구들만 검증(정탐/오탐/미탐)할 수 있게 사전에 있는 단어들을 먼저 인스턴스화
    2. NER/Regex으로 추출한 친구들을 검증할 수 있게 인스턴스화
    3. Okt기반으로 POS가 "Noun", "Number", "Email", "URL", "Foreign", "Alpha"인 친구들을 추출(Span 추출 알고리즘)
    """
        
    def _create_instances(self):
        def find_char_idx_in_sentence(str_find_start_idx, sentence, span_text):
            char_start = sentence.find(span_text, str_find_start_idx)

            if char_start == -1: # 만약 문장 내 span_text가 없다면
                print(f"No matched SPAN_TEXT({span_text}) in SENTENCE({sentence})\n\n")
                return None, None, str_find_start_idx
            else: # 만약 문장 내 span_text가 있다면
                char_end = char_start + len(span_text)
                str_find_start_idx = char_end
                return char_start, char_end, str_find_start_idx
            

        def return_char_idx_to_token_idx(char_start, char_end, offset_mapping):
            token_start, token_end = None, None
            
            # offset mapping을 순회하며 각 토큰별 idx와 char_start, char_end를 추출
            for i, (offset_char_start, offset_char_end) in enumerate(offset_mapping):
                if offset_char_start <= char_start < offset_char_end:
                    token_start = i
                if offset_char_start < char_end <= offset_char_end:
                    token_end = i+1
            
            return token_start, token_end
            

        def dictionary_based_extraction(conn, sentence, table_name, offset_mapping):
            nonlocal no_span_skipped, span_truncated_sent_skipped
            
            # 문장에서 "table_name"사전에 있는 모든 (SPAN_TEXT, label) 추출
            extracted_texts = find_words_in_sentence_for_doc(
                conn=conn,
                sentence=sentence,
                table_name=table_name
            )
            
            str_find_start_idx = 0

            for span_text, label in extracted_texts:
                # sentence내 SPAN_TEXT의 char idx(start, end)를 추출
                char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                    str_find_start_idx=str_find_start_idx,
                    sentence=sentence,
                    span_text=span_text
                )

                # 문장 내 SPAN_TEXT가 없으므로 진행불가(LOG as "no_span_skipped")
                if char_start == None:
                    no_span_skipped += 1
                    continue

                # char idx(start, end)를 token idx(start, end)로 변환
                token_start, token_end = return_char_idx_to_token_idx(
                    char_start=char_start,
                    char_end=char_end,
                    offset_mapping=offset_mapping
                )

                # "max_length truncation"으로 인해 해당 SPAN_TEXT가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                if (token_start is None) or (token_end is None):
                    span_truncated_sent_skipped += 1
                    continue

                # 만약 이전 프로세스 에서 해당 token이 이미 추출된 경우 건너띔
                if is_extracted[token_start] == True and is_extracted[token_end-1] == True:
                    continue

                # 찾은 token idx(start, end)로 해당 SPAN_TEXT 토큰 정보 확인(token_id 및 decoded_token_id)
                span_ids = input_ids[token_start:token_end]
                decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                # 해당 SPAN_TEXT의 label
                label_id = self.label_2_id[label]
                samples_num[label_id] += 1

                # 이후 프로세스에서 중복 추출 방지를 위한 token flag 설정
                for token_idx in range(token_start, token_end):
                    is_extracted[token_idx] = True

                # 인스턴스 추가
                self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 1
                    })
            
        # 로그용
        no_span_skipped = 0
        span_truncated_sent_skipped = 0
        samples_num = [0 for _ in range( len(self.label_2_id) )]

        # DB 접근용 connection
        conn = get_connection()

        # self.samples에 접근하여 검증1, 검증2, 검증3별로 인스턴스 생성하기
        for id, sample_data in tqdm(self.samples.items(), desc="Create Test instances"):
            sentence = sample_data['sentence']

            # 토큰화(tokenizing)
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            # 각 검증단계 별 기 추출된 토큰은 이후 단계에서 추출되면 안되므로 추출 여부 확인용 리스트
            special_ids = set(self.tokenizer.all_special_ids)
            is_extracted = [ token_id in special_ids for token_id in input_ids ]

            ### 1. 사전을 참고하여 사전에 있는 단어만 일단 self.instances에 append
            dictionary_based_extraction(
                conn=conn,
                sentence=sentence,
                table_name="개인정보",
                offset_mapping=offset_mapping
            ) # 개인정보 사전에 있는 단어들 추출하여 self.instances에 append
            dictionary_based_extraction(
                conn=conn,
                sentence=sentence,
                table_name="기밀정보",
                offset_mapping=offset_mapping
            ) # 기밀정보 사전에 있는 단어들 추출하여 self.instances에 append
            dictionary_based_extraction(
                conn=conn,
                sentence=sentence,
                table_name="준식별자",
                offset_mapping=offset_mapping
            ) # 준식별자 사전에 있는 단어들 추출하여 self.instances에 append


            ### 2. NER/REGEX 추출 알고리즘 갖고오기
            # 2-1. Regex 추출 알고리즘
            regex_texts = run_pii_detection(sentence)
            str_find_start_idx = 0
            for span_text in regex_texts:
                # 혜주 : "정규표현식으로 뽑히는 모든것은 개인정보야!" 일동 : "ㅇㅋ"
                label = "개인정보" 
                
                # sentence내 SPAN_TEXT의 char idx(start, end)를 추출
                char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                    str_find_start_idx=str_find_start_idx,
                    sentence=sentence,
                    span_text=span_text
                )

                # 문장 내 SPAN_TEXT가 없으므로 진행불가(LOG as "no_span_skipped")
                if char_start == None:
                    no_span_skipped += 1
                    continue

                # char idx(start, end)를 token idx(start, end)로 변환
                token_start, token_end = return_char_idx_to_token_idx(
                    char_start=char_start,
                    char_end=char_end,
                    offset_mapping=offset_mapping
                )

                # "max_length truncation"으로 인해 해당 SPAN_TEXT가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                if (token_start is None) or (token_end is None):
                    span_truncated_sent_skipped += 1
                    continue

                # 만약 이전 프로세스 에서 해당 token이 이미 추출된 경우 건너띔
                if is_extracted[token_start] == True and is_extracted[token_end-1] == True:
                    continue

                # 찾은 token idx(start, end)로 해당 SPAN_TEXT 토큰 정보 확인(token_id 및 decoded_token_id)
                span_ids = input_ids[token_start:token_end]
                decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                # 해당 SPAN_TEXT의 label
                label_id = self.label_2_id[label]
                samples_num[label_id] += 1

                # 이후 프로세스에서 중복 추출 방지를 위한 token flag 설정
                for token_idx in range(token_start, token_end):
                    is_extracted[token_idx] = True

                # 인스턴스 추가
                self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 2
                    })

            # 2-2. NER 추출 알고리즘
            ner_texts = run_ner(sentence)
            str_find_start_idx = 0
            for span_text in ner_texts:
                # 완철 : "NER로 뽑히는 모든것은 개인정보야!" 일동 : "ㅇㅋ"
                label = "개인정보" 
                
                # sentence내 SPAN_TEXT의 char idx(start, end)를 추출
                char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                    str_find_start_idx=str_find_start_idx,
                    sentence=sentence,
                    span_text=span_text
                )

                # 문장 내 SPAN_TEXT가 없으므로 진행불가(LOG as "no_span_skipped")
                if char_start == None:
                    no_span_skipped += 1
                    continue

                # char idx(start, end)를 token idx(start, end)로 변환
                token_start, token_end = return_char_idx_to_token_idx(
                    char_start=char_start,
                    char_end=char_end,
                    offset_mapping=offset_mapping
                )

                # "max_length truncation"으로 인해 해당 SPAN_TEXT가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                if (token_start is None) or (token_end is None):
                    span_truncated_sent_skipped += 1
                    continue

                # 만약 이전 프로세스 에서 해당 token이 이미 추출된 경우 건너띔
                if is_extracted[token_start] == True and is_extracted[token_end-1] == True:
                    continue

                # 찾은 token idx(start, end)로 해당 SPAN_TEXT 토큰 정보 확인(token_id 및 decoded_token_id)
                span_ids = input_ids[token_start:token_end]
                decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                # 해당 SPAN_TEXT의 label
                label_id = self.label_2_id[label]
                samples_num[label_id] += 1

                # 이후 프로세스에서 중복 추출 방지를 위한 token flag 설정
                for token_idx in range(token_start, token_end):
                    is_extracted[token_idx] = True

                # 인스턴스 추가
                self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 2
                    })
            

            ### 3. Span추출 알고리즘
            # 3-1. 정답지에 있는 친구들 기준으로 우선 추출
            span_texts = find_words_in_sentence_for_doc(
                conn=conn,
                sentence=sentence,
                table_name="정답지",
            )

            # 정답지에서 추출한 친구들을 인스턴스 추출
            str_find_start_idx = 0
            for span_text, label in span_texts:
                if self.is_pii:
                    if label == "기밀정보":
                        print("기밀정보 탐지. 사용 불가능 라벨.")
                        continue
                if not self.is_pii:
                    if label == "준식별자" or label == "개인정보":
                        print("개인정보/준식별자 탐지. 사용 불가능 라벨.")
                        continue
                if label not in self.label_2_id:
                    print(f"사용불가 ({span_text}, {label})")
                    continue

                # sentence내 SPAN_TEXT의 char idx(start, end)를 추출
                char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                    str_find_start_idx=str_find_start_idx,
                    sentence=sentence,
                    span_text=span_text
                )

                # 문장 내 SPAN_TEXT가 없으므로 진행불가(LOG as "no_span_skipped")
                if char_start == None:
                    no_span_skipped += 1
                    continue

                # char idx(start, end)를 token idx(start, end)로 변환
                token_start, token_end = return_char_idx_to_token_idx(
                    char_start=char_start,
                    char_end=char_end,
                    offset_mapping=offset_mapping
                    )
                
                # "max_length truncation"으로 인해 해당 SPAN_TEXT가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                if (token_start is None) or (token_end is None):
                    span_truncated_sent_skipped += 1
                    continue

                # 만약 이전 프로세스에서 해당 token이 이미 추출된 경우 건너띔
                if is_extracted[token_start] == True and is_extracted[token_end-1] == True:
                    continue

                # 찾은 token idx(start, end)로 해당 SPAN_TEXT 토큰 정보 확인(token_id 및 decoded_token_id)
                span_ids = input_ids[token_start:token_end]
                decoded_span_ids = self.tokenizer.convert_ids_to_tokens(span_ids)

                # 해당 SPAN_TEXT의 label
                label_id = self.label_2_id[label]
                samples_num[label_id] += 1

                # 이후 프로세스에서 중복 추출 방지를 위한 token flag 설정
                for token_idx in range(token_start, token_end):
                    is_extracted[token_idx] = True

                # 인스턴스 추가
                self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": span_text,
                        "span_ids": span_ids,
                        "decoded_span_ids": decoded_span_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "label": label_id,
                        "validation_priority": 3
                    })



            # 3-2. okt를 이용한 Span후보가 될 수 있는 POS tag 목록
            target_pos = {"Noun", "Number", "Email", "URL", "Foreign", "Alpha"}
            POS_in_sentence = self.okt.pos( sentence )

            str_find_start_idx = 0

            # 다음 loop에서 사용할 이전 loop 정보
            PRE_LOOP_SPAN_TEXT = ''
            PRE_LOOP_SPAN_POS = '' 
            PRE_LOOP_TOKEN_START = None
            PRE_LOOP_TOKEN_END = None
            PRE_LOOP_SPAN_IDs = []
            PRE_LOOP_DECODED_SPAN_IDs = []

            for span_text, pos in POS_in_sentence:
                # 현재 loop에서 사용할 현재 loop정보
                CURR_LOOP_SPAN_TEXT = span_text
                CURR_LOOP_SPAN_POS = pos
                
                if CURR_LOOP_SPAN_POS in target_pos:
                    # sentence내 SPAN_TEXT의 char idx(start, end)를 추출
                    char_start, char_end, str_find_start_idx = find_char_idx_in_sentence(
                        str_find_start_idx=str_find_start_idx,
                        sentence=sentence,
                        span_text=CURR_LOOP_SPAN_TEXT
                    )

                    # 문장 내 SPAN_TEXT가 없으므로 진행불가(LOG as "no_span_skipped")
                    if char_start == None:
                        no_span_skipped += 1
                        continue

                    # char idx(start, end)를 token idx(start, end)로 변환
                    CURR_LOOP_TOKEN_START, CURR_LOOP_TOKEN_END = return_char_idx_to_token_idx(
                        char_start=char_start,
                        char_end=char_end,
                        offset_mapping=offset_mapping
                        )

                    # "max_length truncation"으로 인해 해당 SPAN_TEXT가 잘린경우 진행불가(LOG as "span_truncated_sent_skipped")
                    if (CURR_LOOP_TOKEN_START is None) or (CURR_LOOP_TOKEN_END is None):
                        span_truncated_sent_skipped += 1
                        continue

                    # 찾은 token idx(start, end)로 해당 SPAN_TEXT 토큰 정보 확인(token_id 및 decoded_token_id)
                    CURR_LOOP_SPAN_IDs = input_ids[CURR_LOOP_TOKEN_START:CURR_LOOP_TOKEN_END]
                    CURR_LOOP_DECODED_SPAN_IDs = self.tokenizer.convert_ids_to_tokens(CURR_LOOP_SPAN_IDs)

                    # 만약 이전 프로세스에서 해당 token이 이미 추출된 경우 건너띔
                    if is_extracted[CURR_LOOP_TOKEN_START] == True and is_extracted[CURR_LOOP_TOKEN_END-1] == True:
                        continue

                    # 만약 CURR_LOOP_DECODED_SPAN_IDs[0]이 ## prefix로 시작하는 경우, pre_loop_token과 합쳐야 하므로
                    if CURR_LOOP_DECODED_SPAN_IDs[0].startswith("##") and PRE_LOOP_TOKEN_END == CURR_LOOP_TOKEN_START:
                        CURR_LOOP_SPAN_TEXT = PRE_LOOP_SPAN_TEXT + CURR_LOOP_SPAN_TEXT
                        CURR_LOOP_SPAN_IDs = PRE_LOOP_SPAN_IDs + CURR_LOOP_SPAN_IDs
                        CURR_LOOP_DECODED_SPAN_IDs = PRE_LOOP_DECODED_SPAN_IDs + CURR_LOOP_DECODED_SPAN_IDs
                        CURR_LOOP_TOKEN_START = PRE_LOOP_TOKEN_START

                    samples_num[0] += 1

                    self.instances.append({
                        "sentence": sentence,
                        "input_ids": input_ids,
                        "decoded_input_ids": decoded_input_ids,
                        "attention_mask": attention_mask,
                        "span_text": CURR_LOOP_SPAN_TEXT,
                        "span_ids": CURR_LOOP_SPAN_IDs,
                        "decoded_span_ids": CURR_LOOP_DECODED_SPAN_IDs,
                        "token_start": CURR_LOOP_TOKEN_START,
                        "token_end": CURR_LOOP_TOKEN_END,
                        "label": self.label_2_id["일반정보"],
                        "validation_priority": 3
                    })
                    
                    # 현재 loop의 정보를 저장
                    PRE_LOOP_SPAN_TEXT = CURR_LOOP_SPAN_TEXT
                    PRE_LOOP_SPAN_POS = CURR_LOOP_SPAN_POS
                    PRE_LOOP_TOKEN_START = CURR_LOOP_TOKEN_START
                    PRE_LOOP_TOKEN_END = CURR_LOOP_TOKEN_END
                    PRE_LOOP_SPAN_IDs = CURR_LOOP_SPAN_IDs
                    PRE_LOOP_DECODED_SPAN_IDs = CURR_LOOP_DECODED_SPAN_IDs

        
        print(f"[Total instances]\nLabel & ids : {self.label_2_id}\nnums : {samples_num}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        instance_df = pd.DataFrame(self.instances)
        instance_df.to_csv(f'../DatasetInstanceSamples/{self.test_name}_test_dataset_samples.csv', index=False)
        
        conn.close()


    def edit_validation_priority(self, idx, edit_to):
        self.instances[idx]["validation_priority"] = edit_to


    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "idx": torch.tensor(idx),
            "sentence": item['sentence'],
            "input_ids": torch.tensor(item["input_ids"]),
            "decoded_input_ids": item["decoded_input_ids"],
            "attention_mask": torch.tensor(item["attention_mask"]),
            "span_text": item["span_text"],
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "label": torch.tensor(item["label"]),
            "validation_priority": item["validation_priority"]
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
    all_data = {"data": []}
    
    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            with open(os.path.join(json_dir, file_name), "r", encoding='utf-8') as f:
                json_file = json.load(f)
                all_data["data"].append( json_file["data"] )
    
    return all_data