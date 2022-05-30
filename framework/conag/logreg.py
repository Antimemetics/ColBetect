import torch
import torch.nn as nn
import torch.nn.functional as F

"""a simple fully connected layer """


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        mid_num1 = int(ft_in / 2)
        # mid_num2 = int(mid_num1/2)

        self.fc = nn.Linear(ft_in, mid_num1)
        self.fc2 = nn.Linear(mid_num1, nb_classes)
        # self.fc3 = nn.Linear(mid_num2, nb_classes)

        for m in self.modules():
            weights_init(m)

    def forward(self, seq):
        ret = self.fc(seq)
        ret = torch.tanh(ret)
        # ret = relu(ret)
        ret = self.fc2(ret)
        # ret2 = torch.tanh(ret)
        # ret3 = self.fc3(ret)
        # ret = torch.sigmoid(ret)
        return ret
