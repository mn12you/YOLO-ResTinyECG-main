import math
import torch
import torch.nn as nn
from torchsummary import summary

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        # input_data = nn.Conv2d(input_data, (3, 3, 3, 16))
        # input_data = nn.MaxPool2d(2, 2, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 16, 32))
        # input_data = nn.MaxPool2d(2, 2, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 32, 64))
        # input_data = nn.MaxPool2d(2, 2, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 64, 128))
        # input_data = nn.MaxPool2d(2, 2, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 128, 256))
        # route_1 = input_data
        # input_data = nn.MaxPool2d(2, 2, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 256, 512))
        # input_data = nn.MaxPool2d(2, 1, 'same')(input_data)
        # input_data = nn.Conv2d(input_data, (3, 3, 512, 1024))

        kernel_size = 3
        stride = 2

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding = 1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=stride),
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding = 1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=stride),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding = 1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=stride),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, padding = 1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=stride),
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=1, padding = 1, bias=False)
            )      

        self.stage2 = nn.Sequential(            
            nn.MaxPool2d(kernel_size=2, stride=stride),
            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=1, padding = 1, bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=1),
            # nn.Conv2d(512, 1024, kernel_size=kernel_size, stride=1, padding = 1, bias=False)            
            )
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv = nn.Conv2d(512, 1024, kernel_size=kernel_size, stride=1, padding = 1, bias=False) 


    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)  
        # C2 = self.maxpool(C2) 
        C2 = self.conv(C2)       

        return C1, C2

if __name__ == "__main__":
    num_blocks = [1,2,8,8,4]
    model = Darknet53()
    print(model)
    test_data = torch.rand(1, 3, 320, 320)
    test_outputs = model(test_data)
    for out in test_outputs:
        print(out.size())
