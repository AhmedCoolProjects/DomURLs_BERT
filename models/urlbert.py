import torch.nn as nn
import torch.nn.functional as F
import torch

from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
)

from .plm import MetaMLP

import os

# Get the path of the current file
file_path = os.path.abspath(__file__)

# Get the directory containing the file
directory_path = os.path.dirname(file_path)

config_kwargs = {
    "cache_dir": None,
    "revision": 'main',
    "use_auth_token": None,
    "hidden_dropout_prob": 0.2,
    "vocab_size": 5000,
}

class URLBertForSequenceClassification(nn.Module):
    def __init__(self, output_size, drop_prob=0.2, pretrained_path=None,
                 meta_dim=0, meta_hidden_dim=128, meta_out_dim=64):
        super(URLBertForSequenceClassification, self).__init__()
        config = AutoConfig.from_pretrained(f"{directory_path}/urlbert_model/", **config_kwargs)


        self.bert_model = AutoModelForMaskedLM.from_config(
            config=config,
        )
        self.bert_model.resize_token_embeddings(config_kwargs["vocab_size"])

        bert_dict = torch.load(f"{directory_path}/urlbert_model/urlBERT.pt", map_location='cpu')

        self.bert_model.load_state_dict(bert_dict)

        self.dropout = nn.Dropout(p=drop_prob)

        self.meta_dim = meta_dim
        if meta_dim > 0:
            self.meta_mlp = MetaMLP(meta_dim, meta_hidden_dim, meta_out_dim, drop_prob)
            clf_in = 768 + meta_out_dim
        else:
            self.meta_mlp = None
            clf_in = 768
        self.classifier = nn.Linear(clf_in, output_size)

    def forward(self, input_ids=None, attention_mask=None, meta=None):

        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1][:,0,:]
        out = self.dropout(hidden_states)

        if self.meta_mlp is not None and meta is not None:
            meta_emb = self.meta_mlp(meta)
            out = torch.cat([out, meta_emb], dim=-1)

        return self.classifier(out)