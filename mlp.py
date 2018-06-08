import math

import torch.nn as nn
import torch.nn.functional as F

from ops import linear


class MLP(nn.Module):
    def __init__(self, num_classes=1000):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # when you add the convolution and batch norm, below will be useful
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):

        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(x, inplace=True)

        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        end_points = {'Predictions': F.softmax(input=x, dim=-1)}

        return x, end_points


def MLPNet(**kwargs):
    model = MLP(**kwargs)
    return model
