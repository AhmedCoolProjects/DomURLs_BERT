import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MetaMLP(nn.Module):
    """Small MLP that embeds the scaled metadata vector before fusion."""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, drop_prob=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PLMEncoder(nn.Module):
    def __init__(self, output_size, drop_prob=0.2,
                 pretrained_path='amahdaouy/BERT_DOMURLS',
                 meta_dim=0, meta_hidden_dim=128, meta_out_dim=64):
        super(PLMEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.dropout = nn.Dropout(drop_prob)

        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_mlp = MetaMLP(meta_dim, meta_hidden_dim, meta_out_dim, drop_prob)
            clf_in = self.transformer.config.hidden_size + meta_out_dim
        else:
            self.meta_mlp = None
            clf_in = self.transformer.config.hidden_size
        self.Classifier = nn.Linear(clf_in, output_size)

    def forward(self, input_ids=None, attention_mask=None, meta=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs[1]
        pooled = self.dropout(pooled)

        if self.meta_mlp is not None and meta is not None:
            meta_emb = self.meta_mlp(meta)
            pooled = torch.cat([pooled, meta_emb], dim=-1)

        return self.Classifier(pooled)
