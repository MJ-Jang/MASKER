import torch
import torch.nn as nn

from transformers import DistilBertModel, DistilBertTokenizer, DistilBertPreTrainedModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_backbone(name, output_attentions=False):
    if name == 'bert':
        from transformers import BertModel, BertTokenizer
        backbone = BertModel.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
    elif name == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer
        backbone = RobertaModel.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.name = 'roberta-base'
    elif name == 'albert':
        from transformers import AlbertModel, AlbertTokenizer
        backbone = AlbertModel.from_pretrained('albert-base-v2', output_attentions=output_attentions)
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        tokenizer.name = 'albert-base-v2'
    elif name == 'distilbert':
        from transformers import DistilBertModel, DistilBertTokenizer
        backbone = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=output_attentions)
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        tokenizer.name = 'distilbert-base-uncased'
    else:
        raise ValueError('No matching backbone network')

    return backbone, tokenizer


class DistilBertBase(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.n_classes = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dense = nn.Linear(config.dim, config.dim)
        self.net_cls = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(self, x):
        attention_mask = (x > 0).long()  # 0 is the pad_token for DistilBERT
        outputs = self.distilbert(x, attention_mask)  # hidden, pooled

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.dense(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.net_cls(pooled_output)  # (bs, num_labels)
        return logits


class BaseNet(nn.Module):
    """ Base network """

    def __init__(self, backbone_name, backbone, n_classes):
        super(BaseNet, self).__init__()
        self.backbone_name = backbone_name
        self.backbone = backbone
        self.n_classes = n_classes
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(768,768)
        self.net_cls = nn.Linear(768, n_classes)  # classification layer

    def forward(self, x):
        if self.backbone_name in ['bert', 'albert']:
            attention_mask = (x > 0).float() # 0 is the pad_token for BERT, AlBERT
            outputs = self.backbone(x, attention_mask)  # hidden, pooled
            outputs = outputs[1]
            out_p = self.dropout(outputs)
            out_cls = self.net_cls(out_p)
            return out_cls

        elif self.backbone_name in ['distilbert']:
            # attention_mask = (x > 0).float() # 0 is the pad_token for DistilBERT
            attention_mask = (x > 0).long() # 0 is the pad_token for DistilBERT
            outputs = self.backbone(x, attention_mask)  # hidden, pooled

            hidden_state = outputs[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)
            pooled_output = self.dense(pooled_output)  # (bs, dim)
            pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
            pooled_output = self.dropout(pooled_output)  # (bs, dim)
            logits = self.net_cls(pooled_output)  # (bs, num_labels)
            return logits

        elif self.backbone_name in ['roberta']:
            attention_mask = (x != 1).float() # 1 is the pad_token for RoBERTa
            out = self.backbone(x, attention_mask)[0]
            out_cls = out[:, 0, :] # take cls token (<s>)
            out_cls = self.dropout(out_cls)
            out_cls = self.dense(out_cls)
            out_cls = torch.tanh(out_cls)
            out_cls = self.dropout(out_cls)
            out_cls = self.net_cls(out_cls)
            return out_cls


class MaskerNet(nn.Module):
    """ Makser network """

    def __init__(self, backbone_name, backbone, n_classes, vocab_size):
        super(MaskerNet, self).__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name
        self.dropout = nn.Dropout(0.1)
        self.n_classes = n_classes
        self.vocab_size = vocab_size

        self.dense = nn.Linear(768,768)
        self.net_cls = nn.Linear(768, n_classes)  # classification layer
        self.net_ssl = nn.Sequential(  # self-supervision layer
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, vocab_size),
        )

    def forward(self, x, training=False):
        if training:  # training mode
            x_orig, x_mask, x_ood = x.chunk(3, dim=1)  # (original, masked, outlier)
            if self.backbone_name in ['bert', 'albert', 'distilbert']:
                # attention_mask = (x_orig > 0).float()
                attention_mask = (x_orig > 0).long()
            elif self.backbone_name in ['roberta']:
                attention_mask = (x_orig != 1).float()

            if self.backbone_name in ['bert', 'albert']:
                out_cls = self.backbone(x_orig, attention_mask)[1]  # pooled feature
                out_cls = self.dropout(out_cls)
                out_cls = self.net_cls(out_cls)  # classification
            elif self.backbone_name in ['roberta']:
                out = self.backbone(x_orig, attention_mask)[0]
                out_cls = out[:, 0, :] # take cls token (<s>)
                out_cls = self.dropout(out_cls)
                out_cls = self.dense(out_cls)
                out_cls = torch.tanh(out_cls)
                out_cls = self.dropout(out_cls)
                out_cls = self.net_cls(out_cls)

            elif self.backbone_name in ['distilbert']:
                outputs = self.backbone(x_orig, attention_mask)  # hidden, pooled
                hidden_state = outputs[0]  # (bs, seq_len, dim)
                pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.dense(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                pooled_output = self.dropout(pooled_output)  # (bs, dim)
                out_cls = self.net_cls(pooled_output)  # (bs, num_labels)

            out_ssl = self.backbone(x_mask, attention_mask)[0]  # hidden feature
            out_ssl = self.dropout(out_ssl)
            out_ssl = self.net_ssl(out_ssl)  # self-supervision

            if self.backbone_name in ['bert', 'albert']:
                out_ood = self.backbone(x_ood, attention_mask)[1]  # pooled feature
                out_ood = self.dropout(out_ood)
                out_ood = self.net_cls(out_ood)  # classification (outlier)

            elif self.backbone_name in ['distilbert']:
                outputs = self.backbone(x_ood, attention_mask)  # hidden, pooled
                hidden_state = outputs[0]  # (bs, seq_len, dim)
                pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.dense(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                pooled_output = self.dropout(pooled_output)  # (bs, dim)
                out_ood = self.net_cls(pooled_output)  # (bs, num_labels)

            elif self.backbone_name in ['roberta']:
                out = self.backbone(x_ood, attention_mask)[0]
                out_ood = out[:, 0, :] # take cls token (<s>)
                out_ood = self.dropout(out_ood)
                out_ood = self.dense(out_ood)
                out_ood = torch.tanh(out_ood)
                out_ood = self.dropout(out_ood)
                out_ood = self.net_cls(out_ood)

            return out_cls, out_ssl, out_ood

        else:  # inference mode
            if self.backbone_name in ['bert', 'albert']:
                attention_mask = (x > 0).float()
                out_cls = self.backbone(x, attention_mask)[1]  # pooled feature
                out_cls = self.dropout(out_cls)
                out_cls = self.net_cls(out_cls)  # classification
            elif self.backbone_name in ['roberta']:
                attention_mask = (x != 1).float()
                out = self.backbone(x, attention_mask)[0]
                out_cls = out[:, 0, :] # take cls token (<s>)
                out_cls = self.dropout(out_cls)
                out_cls = self.dense(out_cls)
                out_cls = torch.tanh(out_cls)
                out_cls = self.dropout(out_cls)
                out_cls = self.net_cls(out_cls)
            elif self.backbone_name in ['distilbert']:
                attention_mask = (x > 0).float()  # 0 is the pad_token for BERT, AlBERT
                outputs = self.backbone(x, attention_mask)  # hidden, pooled

                hidden_state = outputs[0]  # (bs, seq_len, dim)
                pooled_output = hidden_state[:, 0]  # (bs, dim)
                pooled_output = self.dense(pooled_output)  # (bs, dim)
                pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
                pooled_output = self.dropout(pooled_output)  # (bs, dim)
                out_cls = self.net_cls(pooled_output)  # (bs, num_labels)
            return out_cls

