import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter


def get_embedding_net(
    model="resnet50", pretrained=False, fine_tuning=True, weights_path=None
):
    assert not ((not pretrained) and fine_tuning)

    print(
        "Load {} model (pretrained: {}, fine-tuning: {}).".format(
            model, pretrained, fine_tuning
        )
    )

    if model == "resnet18":
        m = models.resnet18(pretrained=pretrained)

        if fine_tuning:
            for param in m.parameters():
                param.requires_grad = False
            m.fc = nn.Linear(512, 1000)

    elif model == "resnet50":
        m = models.resnet50(pretrained=pretrained)

        if fine_tuning:
            import copy

            l4 = copy.deepcopy(m.layer4)
            for param in m.parameters():
                param.requires_grad = False
            m.layer4 = l4
            m.fc = nn.Linear(2048, 1000)

    elif model == "resnet152":
        m = models.resnet152(pretrained=pretrained)

        if fine_tuning:
            for param in m.parameters():
                param.requires_grad = False
            m.fc = nn.Linear(2048, 1000)

    elif model == "vgg16":
        m = models.vgg16(pretrained=pretrained)

        if fine_tuning:
            for param in m.parameters():
                param.requires_grad = False
            m.classifier[6] = nn.Linear(4096, 1000)

    else:
        raise ValueError

    if weights_path is not None:
        print("Initialize weights with {}.".format(weights_path))
        m.load_state_dict(torch.load(weights_path))

    return m


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class QuadrupletNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3, x4):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        output4 = self.embedding_net(x4)
        return output1, output2, output3, output4

    def get_embedding(self, x):
        return self.embedding_net(x)


class Arcface(nn.Module):
    def __init__(
        self,
        embedding_net,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
    ):
        super(Arcface, self).__init__()
        self.embedding_net = embedding_net
        self._s = s
        self._easy_margin = easy_margin
        self._weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self._weight)

        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._threshold = math.cos(math.pi - m)
        self._margin = math.sin(math.pi - m) * m

    def forward(self, x, y):
        f = self.embedding_net(x)

        cos = F.linear(F.normalize(f), F.normalize(self._weight))
        sin = torch.sqrt((1.0 - torch.pow(cos, 2)).clamp(0, 1))
        phi = cos * self._cos_m - sin * self._sin_m
        if self._easy_margin:
            phi = torch.where(cos > 0, phi, cos)
        else:
            phi = torch.where(cos > self._threshold, phi, cos - self._margin)

        one_hot = torch.zeros(cos.size(), device="cuda")
        one_hot.scatter_(1, y.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cos)
        output *= self._s

        return output

    def get_embedding(self, x):
        return self.embedding_net(x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="simple")
    args = parser.parse_args()
    print(get_embedding_net(model=args.model))
