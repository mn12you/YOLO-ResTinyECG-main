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
import cv2
from tqdm import tqdm

def representative_dataset_gen():
    for i in range(100):
        # creating fake images
        image = tf.random.normal([1] + list(image_shape))
        yield [image]
#   a = []
#   with open(opt.img,'r') as imgf:
#         imgp = imgf.read().splitlines()        
#         for save_path in tqdm(imgp):  
#             save_path = save_path.replace('\\', '/')    
#             #数据预处理
#             ori_img = cv2.imread(save_path)        
#             res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
#             if opt.GrayScale:
#                 res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)  # convert to gray scale
#                 img = res_img.reshape(1, cfg["height"], cfg["width"], 1)
#             else:
#                 img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
#             img = torch.from_numpy(img.transpose(0,3, 1, 2))
#             img = img.to(device).float() / 255.0 
#             img = img.astype(np.float32)
#         a = np.array(a)       
#         img = tf.data.Dataset.from_tensor_slices(a).batch(1)
#         for i in img.take(64):
#             yield [i]


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
    if opt.GrayScale:
        image_shape = (1, cfg["width"], cfg["height"])
    else:
        image_shape = (3, cfg["width"], cfg["height"])    
    batch_size=cfg["batch_size"]
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    TFLITE_PATH = opt.tfoutput + "_qat.tflite"
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], cfg["backbone"], True, export_onnx = True, imggray = opt.GrayScale, quantize = True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()

    # converter = tf.lite.TFLiteConverter.from_saved_model(TFLITE_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = val_dataloader
    # This enables quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # This ensures that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # These set the input and output tensors to int8
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    quant_model = converter.convert()

    # Save the quantized file
    with open(TFLITE_PATH, "wb") as f:
        f.write(quant_model)
        