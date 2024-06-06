from mindyolo.models.layers import Conv2d
from mindspore import nn, Tensor
import mindspore.common.dtype as mstype

import numpy as np
import pytest

def test_conv2d():
    in_channels = 3
    out_channels = 16
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    group = 1
    has_bias = True
    weight_init = 'normal'
    bias_init = 'zeros'

    # Create input tensor
    input_tensor = Tensor(np.random.randn(1, in_channels, 32, 32), mstype.float32)

    # Initialize both Conv2d instances
    conv_custom = Conv2d(in_channels, out_channels, kernel_size, stride, 'pad', padding, dilation, group, has_bias,
                         weight_init, bias_init)
    conv_original = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 'pad', padding, dilation, group, has_bias,
                              weight_init, bias_init)

    # Set the same weights and biases for both instances
    conv_custom.weight.set_data(conv_original.weight.data)
    if has_bias:
        conv_custom.bias.set_data(conv_original.bias.data)

    # Get the output from both instances
    output_custom = conv_custom(input_tensor)
    output_original = conv_original(input_tensor)

    # Compare outputs
    assert np.allclose(output_custom.asnumpy(), output_original.asnumpy()), "Outputs do not match!"


if __name__ == "__main__":
    pytest.main([__file__])