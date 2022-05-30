import torch
import torch.nn as nn
from .logreg import LogReg
from global_object.weight import *
from ..layers.discriminator import Discriminator
from ..layers.gcn import GCN
from ..layers.readout import AvgReadout


class Conag(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(Conag, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)
        # self.logreg = LogReg(logreg_hid_units, 1)

    def forward(self, seq1, seq2, seq3, adj, adj3,
                sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn(seq1, adj, sparse)
        c = self.read(h_1, msk)
        c = self.sigm(c)
        # neg sample 1
        h_2 = self.gcn(seq2, adj, sparse)
        # neg sample 2
        h_3 = self.gcn(seq3, adj3, sparse)
        h_neg = torch.cat((h_2, h_3), 1)

        ret = self.disc(c, h_1, h_neg, samp_bias1, samp_bias2)

        '''
        embeds = h_1.detach()
        idx_train = range(0, int(embeds.shape[1] * train_size))
        idx_train = torch.LongTensor(idx_train)
        train_embs = embeds[0, idx_train]
        ret2 = self.logreg(train_embs)
        # [x, 1]
        ret2_resize = ret2.view(1, ret2.shape[0])
        # print(ret1.shape)
        # print(ret2_resize.shape)
        # print(torch.cat((ret1, ret2_resize), 1))
        # exit()
        ret = torch.cat((ret1, ret2_resize), 1)
        '''
        return ret

    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)
        return h_1.detach(), c.detach()
