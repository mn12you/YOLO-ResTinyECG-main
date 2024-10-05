import torch
import torch.nn as nn
import numpy as np
# from torchsummary import summary

# from torchstat import stat
# from fpn import *
# from backbone.shufflenetv2 import *
# from backbone.SEmbednet import *
# from backbone.ResTinynet import *
# from backbone.ResTinynet_q import *
# from backbone.cspdarknet53_tiny import *
# from backbone.darknet53 import *

from model.fpn import *
from model.backbone.shufflenetv2 import *
from model.backbone.SEmbednet import *
from model.backbone.ResTinynet import *
from model.backbone.ResTinynet_q import *
from model.backbone.cspdarknet53_tiny import *
from model.backbone.darknet53 import *

class Detector(nn.Module):
    def __init__(self, classes, anchor_num, backbone, load_param, export_onnx = False, imggray = False, quantize = False):
        super(Detector, self).__init__()
        convGray = 3
        if imggray:
            convGray = 1

        if backbone.find('dark')>0: # cspdarknet53_tiny  
            out_depth = 72
            # stage_out_channels = [256, 1024]  # Darknet53_tiny  
            stage_out_channels = [512, 1024]  # Darknet53 
            num_blocks = [1,2,8,8,4]      # Darknet53
            self.backbone = Darknet53(num_blocks)  
            
        elif backbone.find('csp')>0: # cspdarknet53_tiny  
            out_depth = 72
            stage_out_channels = [256, 512]  # 320           
            self.backbone = darknet53_tiny(None)  
        
        elif backbone.find('Sh')>0: # Shuffle
            out_depth = 72
            stage_out_channels = [-1, 24, 48, 96, 192]  
            self.backbone = ShuffleNetV2(stage_out_channels, load_param, convGray)   

        elif backbone.find('Se')>0: # SEmbed
            out_depth = 72
            stage_out_channels = [-1, 24, 48, 48, 96]  
            stage_blocks = [3, 2, 1]
            KernalS = [3, 1]
            self.backbone = SEmbednet(stage_out_channels, stage_blocks, KernalS, load_param, convGray)

        elif backbone.find('Re')>0: # ResTiny
            out_depth = 36
            stage_out_channels = [-1, 12, 24, 24, 48]  # 320 
            # out_depth = 45
            # stage_out_channels = [-1, 15, 30, 30, 60]   # 80
            # out_depth = 48
            # stage_out_channels = [-1, 16, 32, 32, 64]   # 160 
            if backbone.find('160')>0:
                ratio = 0.75
                # out_depth = int(out_depth*ratio)
                stage_out_channels[1:] = stage_out_channels[1:]*(ratio * np.ones(len(stage_out_channels)-1))
                stage_out_channels = list(map(int, stage_out_channels))        
            # stage_blocks = [3, 4, 6, 3] # Resnet34
            # stage_blocks = [2, 2, 2, 1] # 160
            stage_blocks = [2, 4, 3, 2] # 320 80    
            if quantize:
                self.backbone = ResTinynet_q(stage_out_channels, stage_blocks, load_param, convGray)
                print("quantized...")
            else:
                self.backbone = ResTinynet(stage_out_channels, stage_blocks, load_param, convGray)

        self.export_onnx = export_onnx        
        self.fpn = LightFPN(stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth)

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        C2, C3 = self.backbone(x)      

        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)
        
        out_reg_2 = self.output_reg_layers(reg_2)
        out_obj_2 = self.output_obj_layers(obj_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_3 = self.output_reg_layers(reg_3)
        out_obj_3 = self.output_obj_layers(obj_3)
        out_cls_3 = self.output_cls_layers(cls_3)
        
        if self.export_onnx:
            out_reg_2 = out_reg_2.sigmoid()
            out_obj_2 = out_obj_2.sigmoid()
            out_cls_2 = F.softmax(out_cls_2, dim = 1)

            out_reg_3 = out_reg_3.sigmoid()
            out_obj_3 = out_obj_3.sigmoid()
            out_cls_3 = F.softmax(out_cls_3, dim = 1)

            print("export onnx ...")
            return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), \
                   torch.cat((out_reg_3, out_obj_3, out_cls_3), 1).permute(0, 2, 3, 1)  

        else:
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3

if __name__ == "__main__":
    convGray = 3
    Height = 320
    Width = 320
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Detector(9, 3, '_Sh', load_param = False, export_onnx = False, imggray=False, quantize = False).to(device)
    summary(model, (convGray, Height, Width))
    # test_data = torch.rand(1, 1, 320, 320)
    # # Check FLOPs MAdd        
    # input_tensor = torch.rand(1, 1, 320, 320).to(device)
    stat(model, (convGray, Height, Width))
    # torch.onnx.export(model,                    #model being run
    #                  test_data,                 # model input (or a tuple for multiple inputs)
    #                  "test.onnx",               # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True)  # whether to execute constant folding for optimization
    


