import math
import torch
import torch.nn as nn
# from torchsummary import summary

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x    

class DarknetBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarknetBlock, self).__init__()
         
        ch_hid = in_channels // 2
        self.conv1 = ConvBN(in_channels, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class Darknet53(nn.Module):
    def __init__(self, num_blocks):
        super(Darknet53, self).__init__()
        self.conv = ConvBN(3, 32, kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(512, num_blocks[4], stride=2)

    def _make_layer(self, ch_in, num_blocks, stride=1):
        layers = [ConvBN(ch_in, ch_in*2, stride=stride, padding=1)]
        for i in range(num_blocks):
            layers.append(DarknetBlock(ch_in * 2))
        return nn.Sequential(*layers) 

    def forward(self, x):
        out = self.conv(x)
        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        return c4, c5

if __name__ == "__main__":
    num_blocks = [1,2,8,8,4]
    model = Darknet53(num_blocks)
    # print(model)
    test_data = torch.rand(1, 3, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
