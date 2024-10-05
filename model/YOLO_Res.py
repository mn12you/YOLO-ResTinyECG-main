import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary
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

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, backbone, load_param = False, export_onnx = False, imggray = False, quantize = False):
        super(Detector, self).__init__()
        convGray = 3
        if imggray:
            convGray = 1

 
            self.out_depth = 36
            self.stage_out_channels = [-1, 12, 24, 24, 48]  # 320 

            if backbone.find('160')>0:
                ratio = 0.75   
                self.stage_out_channels[1:] = self.stage_out_channels[1:]*(ratio * np.ones(len(self.stage_out_channels)-1))
                self.stage_out_channels = list(map(int, self.stage_out_channels))       

        self.stage_blocks = [2, 4, 3, 2] # 320 80   

        # Backbone
        self.inp = self.stage_out_channels[1]         
        self.first_conv = nn.Sequential(
            nn.Conv2d(convGray, self.inp, 3, 2, 0, bias=False),
            nn.BatchNorm2d(self.inp),
            nn.ReLU(inplace=True)
            )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        # Res stage 1 (2 blocks): self.stage1 = self.make_layer(ResidualBlock, self.stage_out_channels[2], self.stage_blocks[0], stride=1)
        self.left_1x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[1], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_1x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[1],  self.stage_out_channels[2], 1, 1, bias=False),
            nn.BatchNorm2d( self.stage_out_channels[2])
            )
        self.left_1x2= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_1x2= nn.Sequential()
        
        # Res stage 2 (4 blocks): self.stage2 = self.make_layer(ResidualBlock, self.stage_out_channels[2], self.stage_blocks[1], stride=2)
        self.left_2x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_2x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2],  self.stage_out_channels[2], 1, 2, bias=False),
            nn.BatchNorm2d( self.stage_out_channels[2])
            )
        self.left_2x2= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_2x2= nn.Sequential()
        self.left_2x3= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_2x3= nn.Sequential()
        self.left_2x4= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[2], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[2]),
            )
        self.shortcut_2x4= nn.Sequential()
        
        # Res stage 3 (3 blocks): self.stage3 = self.make_layer(ResidualBlock, self.stage_out_channels[3], self.stage_blocks[2], stride=2)
        self.left_3x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2], self.stage_out_channels[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            )
        self.shortcut_3x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[2],  self.stage_out_channels[3], 1, 2, bias=False),
            nn.BatchNorm2d( self.stage_out_channels[3])
            )
        self.left_3x2= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            )
        self.shortcut_3x2= nn.Sequential()
        self.left_3x3= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[3]),
            )
        self.shortcut_3x3= nn.Sequential()
        
        # Res stage 4 (2 blocks): self.stage4 = self.make_layer(ResidualBlock, self.stage_out_channels[4], self.stage_blocks[3], stride=2)
        self.left_4x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3], self.stage_out_channels[4], 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[4], self.stage_out_channels[4], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[4]),
            )
        self.shortcut_4x1= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[3],  self.stage_out_channels[4], 1, 2, bias=False),
            nn.BatchNorm2d( self.stage_out_channels[4])
            )
        self.left_4x2= nn.Sequential(
            nn.Conv2d(self.stage_out_channels[4], self.stage_out_channels[4], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.stage_out_channels[4], self.stage_out_channels[4], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[4]),
            )
        self.shortcut_4x2= nn.Sequential()
        
        
        
        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")
        self.export_onnx = export_onnx 

        # LightFPN
        input2_depth = self.stage_out_channels[-2] + self.stage_out_channels[-1]
        input3_depth = self.stage_out_channels[-1]
        out_depth = self.out_depth

        self.conv1x1_2 = nn.Sequential(nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias = False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )

        self.conv1x1_3 = nn.Sequential(nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias = False),
                                       nn.BatchNorm2d(out_depth),
                                       nn.ReLU(inplace=True)
                                       )
        
        self.cls_head_2 = self.DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_2 = self.DWConvblock(input2_depth, out_depth, 5)
        
        self.reg_head_3 = self.DWConvblock(input3_depth, out_depth, 5)
        self.cls_head_3 = self.DWConvblock(input3_depth, out_depth, 5)       

        self.output_reg_layers3 = nn.Conv2d(self.out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers3 = nn.Conv2d(self.out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers3 = nn.Conv2d(self.out_depth, classes, 1, 1, 0, bias=True)

        self.output_reg_layers2 = nn.Conv2d(self.out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers2 = nn.Conv2d(self.out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers2 = nn.Conv2d(self.out_depth, classes, 1, 1, 0, bias=True)

    def make_layer(self, block, channels, num_block, stride):
        strides = [stride] + [1] * (num_block-1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.inp, channels, stride))
            self.inp = channels
        return nn.Sequential(*layers)

    def DWConvblock(self, input_channels, output_channels, size):        
        block =  nn.Sequential(nn.Conv2d(output_channels, output_channels, size, 1, 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    
                                    nn.Conv2d(output_channels, output_channels, size, 1, 2, groups = output_channels, bias = False),
                                    nn.BatchNorm2d(output_channels ),
                                    nn.ReLU(inplace=True),
      
                                    nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias = False),
                                    nn.BatchNorm2d(output_channels),
                                    )
        return block
               

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.maxpool(x)
        out = self.left_1x1(x)
        out += self.shortcut_1x1(x)
        out = F.relu(out)
        out = self.left_1x2(out)
        out += self.shortcut_1x2(out)
        C0 = F.relu(out)    # C0 = self.stage1(x)
        out = self.left_2x1(C0)
        out += self.shortcut_2x1(C0)
        out = F.relu(out)
        out = self.left_2x2(out)
        out += self.shortcut_2x2(out)
        out = F.relu(out)    
        out = self.left_2x3(out)
        out += self.shortcut_2x3(out)
        out = F.relu(out)
        out = self.left_2x4(out)
        out += self.shortcut_2x4(out)
        C1 = F.relu(out)    # C1 = self.stage2(C0)
        out = self.left_3x1(C1)
        out += self.shortcut_3x1(C1)
        out = F.relu(out)
        out = self.left_3x2(out)
        out += self.shortcut_3x2(out)
        out = F.relu(out)    
        out = self.left_3x3(out)
        out += self.shortcut_3x3(out)
        C2 = F.relu(out)    # C2 = self.stage3(C1)
        out = self.left_4x1(C2)
        out += self.shortcut_4x1(C2)
        out = F.relu(out)
        out = self.left_4x2(out)
        out += self.shortcut_4x2(out)
        C3 = F.relu(out)   # C3 = self.stage4(C2)

        S3 = self.conv1x1_3(C3)
        cls_3 = self.cls_head_3(S3)
        obj_3 = cls_3
        reg_3 = self.reg_head_3(S3)

        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2),1)
        S2 = self.conv1x1_2(P2)
        cls_2 = self.cls_head_2(S2)
        obj_2 = cls_2
        reg_2 = self.reg_head_2(S2) 
        
        out_reg_2 = self.output_reg_layers2(reg_2)
        out_obj_2 = self.output_obj_layers2(obj_2)
        out_cls_2 = self.output_cls_layers2(cls_2)

        out_reg_3 = self.output_reg_layers3(reg_3)
        out_obj_3 = self.output_obj_layers3(obj_3)
        out_cls_3 = self.output_cls_layers3(cls_3)
        
        if self.export_onnx:
            # out_reg_2 = out_reg_2.sigmoid()
            # out_obj_2 = out_obj_2.sigmoid()
            out_cls_2 = F.softmax(out_cls_2, dim = 1)

            # out_reg_3 = out_reg_3.sigmoid()
            # out_obj_3 = out_obj_3.sigmoid()
            out_cls_3 = F.softmax(out_cls_3, dim = 1)

            print("export onnx ...")
            return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), \
                   torch.cat((out_reg_3, out_obj_3, out_cls_3), 1).permute(0, 2, 3, 1)  

        else:
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3
        
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
    model = Detector(9, 3, '_Re160', load_param = False, export_onnx = False, imggray=True)
    summary(model, (1, 160, 160))
    test_data = torch.rand(1, 1, 160, 160)
    # Check FLOPs MAdd        
    stat(model, (1, 160, 160))
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "test.onnx",               # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
    
    


