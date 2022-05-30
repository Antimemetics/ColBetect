import torch
import torch.nn as nn


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    @staticmethod
    def forward(seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)
