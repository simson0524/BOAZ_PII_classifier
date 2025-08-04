# dataset.py

from torch.utils.data import Dataset
import torch

LABEL_MAP = {
    "일반": 0,
    "개인정보": 1,
    "기밀정보": 2
}

class PIISpanDataset(Dataset): # 수정 요망
    def __init__(self, data, tokenizer, min_span_len=1, max_span_len=10, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.min_span_len = min_span_len
        self.max_span_len = max_span_len
        self.max_length = max_length

    def generate_span_candidates(self, seq_len):
        """Generate span candidates(represented as index)

        Args:
            seq_len (int): Length of sequence

        Returns:
            List(tuple): Candidates of spans
        """
        span_candidates = []
        for start in range( seq_len ):
            for end in range(min(start+self.min_span_len, seq_len), min(start+self.max_span_len, seq_len)):
                span_candidates.append( (start, end) )
        return span_candidates
        # 전화번호나, 주민등록번호 등 긴 번호로 나타나는 PII는 
        # tokenize결과에 따라 너무 긴 경우 후보 span len를 길게 조정 -> 연산부담 이므로,
        # 정규표현식 적용할 수 있는 함수 추가 및 반영방법 고민 요망 
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text, gt_spans = item['text'], set(map(tuple, item['spans'])) # 실제 라벨링 방식에 따라 수정하기 - 1
        encoding = self.tokenizer(text, 
                                  truncation=True, 
                                  padding='max_length', 
                                  max_length=self.max_length,
                                  return_tensors='pt', 
                                  return_offsets_mapping=True)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        seq_len = (attention_mask == 1).sum().item()

        # Generate span candidates
        span_candidates = self.generate_span_candidates(seq_len)
        span_labels = [1 if span in gt_spans else 0 for span in span_candidates] # 실제 라벨링 방식에 따라 수정하기 - 2

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "span_indices": span_candidates,
                "span_labels": torch.tensor(span_labels, dtype=torch.float)}
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    span_indices = [b['span_indices'] for b in batch]
    span_labels = [b['span_labels'] for b in batch]

    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "span_indices": span_indices,
            "span_labels": span_labels}