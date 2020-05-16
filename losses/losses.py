import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin=3.0):
        super(TripletLoss, self).__init__()
        self._margin = margin
        print("Triplet Loss initialized with marigin {}.".format(margin))

    def forward(self, anc, pos, neg, size_average=True):
        d_ap = (anc - pos).pow(2).sum(1)
        d_an = (anc - neg).pow(2).sum(1)

        loss = F.relu(d_ap - d_an + self._margin)
        return loss.mean() if size_average else loss.sum()


class QuadrupletLoss(nn.Module):
    """Implementation of https://arxiv.org/abs/1704.01719."""

    def __init__(self, a1=1.0, a2=1.0):
        super(QuadrupletLoss, self).__init__()
        self._a1 = a1
        self._a2 = a2

    def forward(self, anc, pos, neg1, neg2, size_average=True):
        d_ap = (anc - pos).pow(2).sum(1)
        d_an = (anc - neg1).pow(2).sum(1)
        d_nn = (neg1 - neg2).pow(2).sum(1)

        loss = F.relu(d_ap - d_an + self._a1) + F.relu(d_ap - d_nn + self._a2)
        return loss.mean() if size_average else loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        logp = self.ce(x, y)
        p = torch.exp(-logp)
        loss = (1 - p) ** self._gamma * logp
        return loss.mean()
