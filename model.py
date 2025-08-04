# PIIClassifier/model.py

import torch.nn as nn
import torch

class SpanPIIClassifier(nn.Module):
    def __init__(self, pretrained_bert, num_labels=4):
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.hidden_size = pretrained_bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask, token_start, token_end):
        outputs = self.pretrained_bert(
            input_ids=input_ids, # shape: (batch_size, token_len)
            attention_mask=attention_mask # shape: (batch_size, token_len)
            )        
        last_hidden = outputs.last_hidden_state # shape: (batch_size, token_len, hidden_size), embeddings of each token

        # Batch wise 인덱싱
        batch_size = input_ids.size(0)
        start_embeds = last_hidden[torch.arange(batch_size), token_start] # shape: (batch_size, hidden_size)
        end_embeds = last_hidden[torch.arange(batch_size), token_end] # shape: (batch_size, hidden_size)

        # Concat start+end hidden
        span_representation = torch.cat([start_embeds, end_embeds], dim=-1) # shape: (batch_size, hidden_size*2)

        # Classification
        logits = self.classifier(span_representation)

        return {"logits": logits} 