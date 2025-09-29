# PIIClassifier/model.py

import torch.nn as nn
import torch
import torch.nn.functional as F


class SpanPIIClassifier(nn.Module):
    def __init__(self, pretrained_bert, num_labels=4, use_focal=True, alpha=1.0, gamma=2.0, reduction="mean"):        
        super().__init__()
        self.pretrained_bert = pretrained_bert
        self.hidden_size = pretrained_bert.config.hidden_size
        self.num_labels = num_labels
    

        # Focal loss 관련 하이퍼파라미터
        self.use_focal = use_focal
        self.alpha = alpha #희귀 클래서의 주는 가중치
        self.gamma = gamma #샘플 난이도 조절용. 정답 확률이 높은 쉬운 샘플은 무시하고, 정답 확률이 낮은 어려운 샘플은 강조하게 됨.
        self.reduction = reduction
        

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_labels)
        )

    # TODO : Focal Loss ;;;
    def focal_loss(self, logits, labels):
        """
        logits: (batch_size, num_labels)
        labels: (batch_size,)
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")  # 기본 CE
        pt = torch.exp(-ce_loss)  # 확률값
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def forward(self, input_ids, attention_mask, token_start, token_end,labels = None):
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

        loss = None
        if labels is not None:
            if self.use_focal:
                loss = self.focal_loss(logits, labels)
            else:
                loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": loss} 