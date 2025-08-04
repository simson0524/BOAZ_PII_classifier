# model.py

import torch.nn as nn
import torch

class SpanPIIClassifier(nn.Module):
    def __init__(self, pretrained_bert, hidden_size=768):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1) # binary classification -> 4개로 수정요망
        )

    def forward(self, input_ids, attention_mask, span_indices_batch):
        outputs = self.pretrained_bert(
            input_ids=input_ids, # "input_ids" shape: (batch_size, token_len)
            attention_mask=attention_mask
        ) # "BERT output" shape: (batch_size, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state # shape: (batch_size, token_len, hidden_size), embeddings of each token
        ##### skt/kobert-base-v1에는 없는 매서드일수도

        batch_span_logits = []

        for batch in range( len(span_indices_batch) ):
            span_indices = span_indices_batch[ batch ]
            span_reps = []
            for (start, end) in span_indices:
                hidden_start = last_hidden[batch, start]
                hidden_end = last_hidden[batch, end]
                span_rep = torch.cat([hidden_start, hidden_end])
                span_reps.append( span_rep )
            span_reps = torch.stack( span_reps ) # shape: (num_spans, hidden_size*2)
            logits = self.classifier( span_reps ).squeeze(-1) # shape: (num_spans, )
            batch_span_logits.append( logits )

        return batch_span_logits 