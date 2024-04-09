import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, embeds_dim, p=0.1):
        super().__init__()
        self.embeds_dim = embeds_dim
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.dropout = nn.Dropout(p)
        self.linear_heads = nn.ModuleList([nn.Linear(embeds_dim, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(embeds_dim, 2, bias=True))
        self.loss_func = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, inputs_embeds):
        batch_size, seq_len, dim = inputs_embeds.size()
        assert dim == self.embeds_dim
        pooler = nn.AvgPool1d(seq_len)
        cls_hidden = pooler(inputs_embeds.permute(0, 2, 1)).permute(0, 2, 1).squeeze(1)
        cls_hidden = self.dropout(cls_hidden)

        predictions = []
        for i in range(14):
            predictions.append(self.linear_heads[i](cls_hidden).argmax(dim=1))
        return torch.stack(predictions, dim=1)

    def get_multilabel_loss(self, out, labels):
        # labels = labels.permute(1, 0)
        batch_size = len(out)
        batch_loss = 0.0
        labels = labels.float()
        out = out.float()
        for j in range(len(out)):
            batch_loss += self.loss_func(out[j], labels[j])
        loss = batch_loss / batch_size
        return loss
