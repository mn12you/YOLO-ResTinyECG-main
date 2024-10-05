import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
# from torchstat import stat


class ResidualBlock(nn.Module):
    def __init__(self, inp, outp, stride=1):
        super(ResidualBlock, self).__init__()

        self.left= nn.Sequential(
            nn.Conv2d(inp, outp, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.ReLU(inplace=True),
            nn.Conv2d(outp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            )
        self.shortcut= nn.Sequential()
        if stride != 1 or inp != outp:
            self.shortcut= nn.Sequential(
            nn.Conv2d(inp, outp, 1, stride, bias=False),
            nn.BatchNorm2d(outp)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResTinynet(nn.Module):
    def __init__(self, stage_out_channels, stage_blocks, load_param, convGray):
        super(ResTinynet, self).__init__()

        self.inp = stage_out_channels[1]

        # building first layer
        # self.first_conv = nn.Sequential(
        #     nn.Conv2d(3, self.inp, 3, 2, 0, bias=False),
        #     nn.BatchNorm2d(self.inp),
        #     nn.ReLU(inplace=True)
        #     )
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.first_conv = nn.Sequential(
            nn.Conv2d(convGray, self.inp, 3, 2, 0, bias=False),
            nn.BatchNorm2d(self.inp),
            nn.ReLU(inplace=True)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.stage1 = self.make_layer(ResidualBlock, stage_out_channels[2], stage_blocks[0], stride=1)
        self.stage2 = self.make_layer(ResidualBlock, stage_out_channels[2], stage_blocks[1], stride=2)
        self.stage3 = self.make_layer(ResidualBlock, stage_out_channels[3], stage_blocks[2], stride=2)
        self.stage4 = self.make_layer(ResidualBlock, stage_out_channels[4], stage_blocks[3], stride=2)
        
        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def make_layer(self, block, channels, num_block, stride):
        strides = [stride] + [1] * (num_block-1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.inp, channels, stride))
            self.inp = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.maxpool(x)
        C0 = self.stage1(x)
        C1 = self.stage2(C0)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3

    def _initialize_weights(self):        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(tensor, gain=1)
                # torch.nn.init.xavier_normal_(tensor, gain=1)
                # torch.nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # torch.nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
                # nn.init.normal_(m, mean=0, std=0.33)
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    # from torchstat import stat
    import numpy as np
    convGray=1
    Height = 320
    Width = 320
    stage_out_channels = [-1, 12, 24, 24, 48]  # 320 

    backbone = 'Re160'
            
    if backbone.find('160')>0:        
        Height = 160
        Width = 160
        ratio = 0.75
        # out_depth = int(out_depth*ratio)
        stage_out_channels[1:] = stage_out_channels[1:]*(ratio * np.ones(len(stage_out_channels)-1))
        stage_out_channels = list(map(int, stage_out_channels))   
  
    if backbone.find('34')>0:
        # convGray=3
        stage_blocks = [3, 4, 6, 3] # Resnet34   
    else:  
        stage_blocks = [2, 4, 3, 2] # Re Re160  
        print(stage_blocks)

    model = ResTinynet(stage_out_channels, stage_blocks, load_param = False, convGray=convGray)
    print(model)
    test_data = torch.rand(1, convGray, Height, Width)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())

    # Check FLOPs MAdd        
    stat(model, (convGray, Height, Width))