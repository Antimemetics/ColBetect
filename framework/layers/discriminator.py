import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Bilinear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        # x1 & x2: [batch_size, n_h]
        # out_features = 1

        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            weights_init(m)

    def forward(self, c, h_pl, h_anomaly, s_bias1=None, s_bias2=None):
        # unsqueeze
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        # h_pl sample
        # h_mi neg sample
        # print(c_x.shape)
        # print(h_pl.shape)
        # print(h_mi.shape)

        c_x_resize = torch.unsqueeze(c, 1)
        c_x_resize = c_x_resize.expand_as(h_anomaly)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        # sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
        sc_3 = torch.squeeze(self.f_k(h_anomaly, c_x_resize), 2)

        # bias
        if s_bias1 is not None:
            sc_1 += s_bias1
        # if s_bias2 is not None:
        # sc_2 += s_bias2

        # logits = torch.cat((sc_1, sc_2), 1)
        logits = torch.cat((sc_1, sc_3), 1)

        return logits
