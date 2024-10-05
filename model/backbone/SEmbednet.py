import torch
import torch.nn as nn
# from torchsummary import summary

class SEmbednet(nn.Module):
    def __init__(self, stage_out_channels, stage_blocks, KernalS, load_param, convGray):
        super(SEmbednet, self).__init__()

        self.stage_repeats = stage_blocks
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(convGray, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):                    
                if i == 0:
                    branch_main = [
                    nn.Conv2d(input_channel, output_channel, KernalS[0], 1, 0, bias=False), 
                    # nn.BatchNorm2d(output_channel),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
                    ]
                    self.branch_main = nn.Sequential(*branch_main)
                    stageSeq.append(self.branch_main)
                else:
                    branch_main = [
                    nn.Conv2d(input_channel, output_channel, KernalS[1], 1, 0, bias=False), 
                    # nn.BatchNorm2d(output_channel),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
                    ]
                    self.branch_main = nn.Sequential(*branch_main)
                    stageSeq.append(self.branch_main)          
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))
        
        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.maxpool(x)
        C1 = self.stage2(x)
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
                # nn.init.normal_(m, mean=0, std=1)
                nn.init.constant_(m.weight, 1)
                # nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = SEmbednet(stage_out_channels = [-1, 24, 48, 48, 96], stage_blocks = [3, 2, 1], KernalS = [3, 1], load_param = False, convGray = 1)
    # print(model)
    test_data = torch.rand(1, 1, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
