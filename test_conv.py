from mindyolo.models.layers.common import Concat
from mindyolo.models.layers import Conv2d
from mindspore import nn, Tensor
import mindspore.common.dtype as mstype

import numpy as np
from mindyolo.models.layers import autopad

import pytest


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        p1 = [32, 3, 2, 'pad', autopad(3, None, 1), 1, 1]
        p2 = [64, 3, 2, 'pad', autopad(3, None, 1), 1, 1]
        p3 = [32, 1, 1, 'pad', autopad(3, None, 1), 1, 1]
        p4 = [32, 1, 1, 'pad', autopad(3, None, 1), 1, 1]
        p5 = [32, 3, 1, 'pad', autopad(3, None, 1), 1, 1]
        p_6 = [1]
        p7 = [32, 3, 1, 'pad', autopad(3, None, 1), 1, 1]

        self.l1 = Conv2d(3, *p1)
        self.l2 = Conv2d(32, *p2)
        self.l3 = Conv2d(64, *p3)
        self.l4 = Conv2d(32, *p4)
        self.l5 = Conv2d(32, *p5)
        self.cat_6 = Concat(*p_6)
        self.l7 = Conv2d(32, *p5)

    def construct(self, x):
        output = []
        o1 = self.l1(x)
        o2 = self.l2(o1)
        o3 = self.l3(o2)
        o4 = self.l4(o2)
        o5 = self.l5(o4)
        o6 = self.cat_6((o2,o3,o4,o5))
        o7 = self.l7(o6)
        output.append(o1)
        output.append(o2)
        output.append(o3)
        output.append(o4)
        output.append(o5)
        output.append(o6)
        output.append(o7)
        return output


def test_conv2d():
    input_tensor = Tensor(np.random.randn(16, 3, 640, 640), mstype.float32)
    net = Net()
    output = net(input_tensor)
    for i, o in enumerate(output):
        print(f"o{i} = ", o.shape())

   

if __name__ == "__main__":
    test_conv2d()