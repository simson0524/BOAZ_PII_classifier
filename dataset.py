# dataset.py

from torch.utils.data import Dataset
import torch
import json
import os

label_2_id = {"일반" : 0, "개인" : 1, "기밀" : 2, "준식별" : 3}
id_2_label = {0 : "일반", 1 : "개인", 2 : "기밀", 3 : "준식별"}

class PIISpanDataset(Dataset): # 수정 요망
    def __init__(self, json_data, tokenizer, max_length=512):
        self.samples = {data['id']: data for data in json_data["data"]}
        self.annotations = {ann['id']: ann['annotations'] for ann in json_data['annotations']}
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.max_length = max_length
        self.instances = []
        self._create_instances()

    def _create_instances(self):
        for sent_id, anns in self.annotations.items():
            sent = self.samples[sent_id]["sentence"]
            encoding = self.tokenizer(
                sent,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding="max_length"
            )
            offset_mapping = encoding.pop("offset_mapping")

            for ann in anns:
                start, end, label = ann['start'], ann['end'], ann['label']
                if label not in self.label_2_id:
                    print(f"[{label}] label is not available")
                    continue
                label_id = self.label_2_id[label]

                # span을 토큰 인덱스로 mapping
                token_start, token_end = None, None
                for idx, (s, e) in enumerate(offset_mapping):
                    if s == start:
                        token_start = idx
                    if e == end:
                        token_end = idx
                
                # span이 잘린 경우 건너뛰기
                if token_start is None or token_end is None:
                    print(f"Current Span is truncated. Invalid Data")
                    continue

                self.instances.append({"input_ids": encoding["input_ids"],
                                       "attention_mask": encoding["attention_mask"],
                                       "token_start": token_start,
                                       "token_end": token_end,
                                       "label": label_id,
                                       "sentence": sent,
                                       "span_text": sent[start:end]})

    def __getitem__(self, idx):
        item = self.instances[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "token_start": torch.tensor(item["token_start"]),
            "token_end": torch.tensor(item["token_end"]),
            "labels": torch.tensor(item["label"]),
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
                all_data["data"].extend( json_file["data"] )
                all_data["annotations"].extend( json_file["annotations"] )
    
    return all_data