import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MetaMLP(nn.Module):
    """Small MLP that embeds the scaled metadata vector before fusion.

    With `use_layernorm`, applies LayerNorm to the output so its scale matches
    BERT's already-LayerNormed [CLS] before concatenation/gating.
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, drop_prob=0.2, use_layernorm=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x))


def _build_head(in_dim, num_classes, head_hidden_dim=0, drop_prob=0.2):
    """Single Linear if head_hidden_dim==0 (default), else a 2-layer MLP head."""
    if head_hidden_dim and head_hidden_dim > 0:
        return nn.Sequential(
            nn.Linear(in_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(head_hidden_dim, num_classes),
        )
    return nn.Linear(in_dim, num_classes)


class PLMEncoder(nn.Module):
    def __init__(self, output_size, drop_prob=0.2,
                 pretrained_path='amahdaouy/BERT_DOMURLS',
                 meta_dim=0, meta_hidden_dim=128, meta_out_dim=64,
                 fusion_mode="concat", head_hidden_dim=0, meta_layernorm=False):
        super(PLMEncoder, self).__init__()
        self.transformer = AutoModel.from_pretrained(pretrained_path)
        self.dropout = nn.Dropout(drop_prob)
        self.fusion_mode = fusion_mode

        hidden_size = self.transformer.config.hidden_size
        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_mlp = MetaMLP(meta_dim, meta_hidden_dim, meta_out_dim, drop_prob,
                                    use_layernorm=meta_layernorm)
            self.gate_proj = (
                nn.Linear(hidden_size, meta_out_dim) if fusion_mode == "gated" else None
            )
            clf_in = hidden_size + meta_out_dim
        else:
            self.meta_mlp = None
            self.gate_proj = None
            clf_in = hidden_size
        self.Classifier = _build_head(clf_in, output_size, head_hidden_dim, drop_prob)

    def forward(self, input_ids=None, attention_mask=None, meta=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs[1]
        pooled = self.dropout(pooled)

        if self.meta_mlp is not None and meta is not None:
            meta_emb = self.meta_mlp(meta)
            if self.gate_proj is not None:                       # gated fusion
                gate = torch.sigmoid(self.gate_proj(pooled))
                meta_emb = gate * meta_emb
            pooled = torch.cat([pooled, meta_emb], dim=-1)

        return self.Classifier(pooled)
