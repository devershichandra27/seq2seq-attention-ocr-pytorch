import torch
from torch import nn as nn

from src.modules.custom import Conv2dSamePadding, BatchNorm2d


class Encoder(nn.Module):
    def __init__(self, image_channels, enc_hidden_size, LOAD_PATH=None):
        super(Encoder, self).__init__()

        self.cnn = nn.Sequential(
            Conv2dSamePadding(image_channels, 64, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d(2, 2),
            Conv2dSamePadding(64, 128, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d(2, 2),
            Conv2dSamePadding(128, 256, 3, 1, 0, bias=False), BatchNorm2d(256), nn.ReLU(True),
            Conv2dSamePadding(256, 256, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            Conv2dSamePadding(256, 512, 3, 1, 0, bias=False), BatchNorm2d(512), nn.ReLU(True),
            Conv2dSamePadding(512, 512, 3, 1, 0, bias=False), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            Conv2dSamePadding(512, 512, 2, 1, 0, bias=False), BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.rnn = nn.LSTM(input_size=512, hidden_size=enc_hidden_size, bidirectional=True)

        if LOAD_PATH is not None:
            self.load_state_dict(torch.load(LOAD_PATH))

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be one. encoder_wpl.py"
        conv = conv.squeeze(2)  # (b, c,w)
        rnn_inp = conv.permute(2, 0, 1)
        encoder_outputs, (h_n, c_n) = self.rnn(rnn_inp)
        return encoder_outputs, (h_n, c_n)
