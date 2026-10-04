import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin: logit = s * cos(theta + m) for the target class."""

    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.s, self.m, self.easy_margin = s, m, easy_margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb, labels):
        cos = F.linear(F.normalize(emb.float()), F.normalize(self.weight.float())).clamp(-1, 1)
        sin = torch.sqrt((1.0 - cos * cos).clamp(min=1e-7))
        phi = cos * self.cos_m - sin * self.sin_m                  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            phi = torch.where(cos > self.th, phi, cos - self.mm)
        onehot = F.one_hot(labels, cos.size(1)).bool()
        return self.s * torch.where(onehot, phi, cos), self.s * cos   # (margin logits, plain logits)
