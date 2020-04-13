import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArcMarginModel(nn.Module):
    def __init__(self, num_classes, emb_size, easy_margin=False, margin_m=0.5, margin_s=64.0):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        #if self.easy_margin:
        #    phi = torch.where(cosine > 0, phi, cosine)
        #else:
        #    phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = cosine * 1.0  # make backward works
        batch_size = len(output)
        output[range(batch_size), label] = phi[range(batch_size), label]
        return output * self.s
        #one_hot = torch.zeros(cosine.size(), device=device)
        #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        #output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        #output *= self.s
        #return output