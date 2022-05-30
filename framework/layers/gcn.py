import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()

        # nn.Linear
        # [batch_size, in_ft] -> [batch_size, out_ft]
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        # PReLU(x) = x (x >= 0) or ax (x < 0)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            weights_init(m)

    # (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        # 1 seq -> 出seq_fts
        seq_fts = self.fc(seq)
        # print('gcn ' + str(seq.shape))
        # print('gcn ' + str(adj.shape))

        # 2 x adj
        if sparse:
            out = torch.unsqueeze(
                torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        # 3 bias
        if self.bias is not None:
            out += self.bias

        # 4 out
        return self.act(out)
