import torch
from torch import nn
from torch.nn import functional as F
from MPNCOV.python import MPNCOV
from senet.se_module import SELayer
from Models.GCNet import ContextBlock
from Models.bam import BAM
from torch.nn.parameter import Parameter
from senet.se_module_updown import SELayer_updown
from Models.eca import eca_layer


def conv1x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class _ASPNet(nn.Module):
    def __init__(self, in_channels):
        super(_ASPNet, self).__init__()
        self.redim = 512
        self.in_channels = in_channels
        self.soca_c = SOCA_C()

        self.bn1_1 = nn.BatchNorm2d(self.redim)
        self.bn1_2 = nn.BatchNorm2d(self.redim)
        self.bn1_3 = nn.BatchNorm2d(self.redim)
        self.conv1_1_1_1 = conv1x1(self.in_channels, self.redim)
        self.conv1_1_1_2 = conv1x1(self.in_channels, self.redim)

        self.se_second = SELayer_updown(self.redim, 2, self.in_channels)
        self.bn = nn.BatchNorm2d(self.redim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.relu_normal = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        # BIM
        out1_c = self.conv1_1_1_1(x)
        out1 = self.bn1_1(out1_c)
        out1 = self.relu(out1)
        out2_c = self.conv1_1_1_2(x)
        out2 = self.bn1_2(out2_c)
        out2 = self.relu(out2)
        innerproduct = torch.mul(out1, out2)
        out = self.bn1_3(innerproduct)
        out = self.relu(out)

        # NCEB
        g_x = out1_c.view(batch_size, out1_c.size(1), -1)
        MPN_X_C = self.soca_c(out2_c)
        f_div_C = F.softmax(MPN_X_C, dim=-1)
        y_c = torch.matmul(f_div_C, g_x)
        y_c = y_c.view(batch_size, y_c.size(1), *x.size()[2:])

        # add
        y = y_c + out

        # CAB
        y = self.se_second(y)
        out = x * y.expand_as(x)
        z = out

        return z


class ASPNet(_ASPNet):
    def __init__(self, in_channels):
        super(ASPNet, self).__init__(in_channels)


# second-order Channel attention (SOCA)
class SOCA_C(nn.Module):
    def __init__(self):
        super(SOCA_C, self).__init__()

    def forward(self, x):
        cov_mat = MPNCOV.CovpoolLayer(x)
        return cov_mat


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:

        img = torch.zeros(2, 3, 20, 20)
        net = ASPNet(3)
        out = net(img)
        print(out.size())



