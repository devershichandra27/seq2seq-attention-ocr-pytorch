from functools import reduce
from operator import __add__

import torch
from torch import nn


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self._zero_pad_2d = nn.ZeroPad2d(
            reduce(__add__, [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

        def forward(self, input):
            input = self._zero_pad_2d(input)
            return self._conv_forward(input, self.weight, self.bias)


class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init()
        shape = (1, num_features, 1, 1)

        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_variance = torch.ones(shape)

    def forward(self, x):
        self.moving_mean = self.moving_mean.type_as(x)
        self.moving_variance = self.moving_variance.type_as(x)

        y = self._batch_norm(x, eps=1e-3, momentum=0.99)
        return y

    def _batch_norm(self, x, eps, momentum):
        gamma = self.weight.type_as(x)
        beta = self.bias.type_as(x)

        if not torch.is_grad_enabled():
            x_hat = (x - self.moving_mean) / torch.sqrt(self.moving_variance + eps)
        else:
            assert (len(x.shape)) == 4
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

            x_hat = (x - mean) / torch.sqrt(var + eps)

            self.moving_mean = momentum * self.moving_mean + (1.0 - momentum) * mean
            self.moving_variance = momentum * self.moving_variance + (1.0 - momentum) * var
        y = gamma * x_hat + beta
        return y
