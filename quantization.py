import argparse

import torch
import model.detector
import utils.utils
from torchstat import stat

import onnx
from onnx_tf.backend import prepare
from onnxsim import simplify
import tensorflow as tf
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from torchsummary import summary
import os
import utils.datasets
import numpy as np


def rep_dataset():
    """Generator function to produce representative dataset for post-training quantization."""
    
    test_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)
    batch_size=cfg["batch_size"]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 训练集
    val_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=False,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )

    # Use a few samples from the training set.
    for _ in range(100):
        img = iter(val_dataloader).next()[0].numpy()
        img = [(img.astype(np.float32))]
        yield img

def quantize_onnx_model(onnx_model_path, quantized_model_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnx
    onnx_opt_model = onnx.load(onnx_model_path)
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)


if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/8class_5s/QRS_AAMI_8_5s_Re160_1.data ', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='./weights/8class_5s_Re160-qF.pth', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--output', type=str, default='./8class_5s_Re160q.onnx', 
                        help='The path where the onnx model is saved')                        
    parser.add_argument('--tfoutput', type=str, default='./8class_5s_Re160qF', 
                        help='The path where the onnx model is saved')
    parser.add_argument('--GrayScale', type=str, default=True, 
                        help='Input image in Gray Scale: False/True')

    num_calibration_batches = 10

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    test_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False)
    batch_size=cfg["batch_size"]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 训练集
    val_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=False,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )

    # torch.cuda.is_available()
    # torch.cuda.device_count()
    # torch.cuda.get_device_name(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], cfg["backbone"], True, export_onnx = True, imggray = opt.GrayScale, quantize = True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()
# Torch --> tflite
    # torch.quantization.convert(model, inplace=False)
    # TFLITE_PATH = opt.tfoutput + "_int8.tflite"
    # torch.backends.quantized.engine = 'fbgemm'
  
    # if opt.GrayScale:
    #     test_data = torch.rand(1, 1, cfg["height"], cfg["width"]).to(device)
    # else:
    #     test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    # converter = TFLiteConverter(model,
    #                             test_data,
    #                             tflite_path=TFLITE_PATH)
    # converter.convert()
# Quantization    
    # torch.backends.quantized.engine = 'fbgemm'
    # model_fp32 = torch.quantization.convert(model, inplace=False)
    TFLITE_PATH = opt.tfoutput + "_qat.tflite"
    
    # # Our initial baseline model which is FP32
    # model_fp32 = model
    # # summary(model_fp32, input_size=(1, cfg["height"], cfg["width"]))
    # # Sets the backend for x86
    # model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # # Prepares the model for the next step i.e. calibration.
    # # Inserts observers in the model that will observe the activation tensors during calibration
    # model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)

    # # Converts the model to a quantized model(int8) 
    # model_quantized = torch.quantization.convert(model_fp32_prepared) # Quantize the model
    # # stat(model_quantized, (1, cfg["height"], cfg["width"]))
    # summary(model_quantized, input_size=(1, cfg["height"], cfg["width"]))
  
    if opt.GrayScale:
        test_data = torch.rand(1, 1, cfg["height"], cfg["width"]).cpu
        # test_data = torch.randn(1, 1,  cfg["height"], cfg["width"], dtype=torch.uint8).to(device)
    else:
        test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).cpu
        # test_data = torch.randn(1, 3,  cfg["height"], cfg["width"], dtype=torch.uint8).to(device)
    
    converter = TFLiteConverter(model,
                                test_data,
                                tflite_path=TFLITE_PATH,                                
                                quantize_target_type='uint8')
    converter.convert()     
    

'''
    # quantization 
    dummy_input = torch.randn(1, 1, 160, 160)        

    quantizer = QATQuantizer(model, dummy_input, work_dir='out', config='fbgemm')
    q_model = quantizer.quantize()

    for _ in range(10):
        dummy_input = torch.randn(1, 1, 160, 160)
        q_model(dummy_input)

    q_model = torch.quantization.convert(q_model)

    print(q_model)

# Torch --> tflite       
    TFLITE_PATH = opt.tfoutput + "_int8.tflite"
    torch.backends.quantized.engine = 'fbgemm'
  
    if opt.GrayScale:
        test_data = torch.rand(1, 1, cfg["height"], cfg["width"]).to(device)
    else:
        test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)

    converter2 = TFLiteConverter(q_model,
                                test_data,
                                tflite_path=TFLITE_PATH)
    converter2.convert()
'''


    

