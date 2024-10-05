import argparse

import torch
import model.detector
import utils.utils
from torchsummary import summary

import onnx
from onnx_tf.backend import prepare
from onnxsim import simplify
import tensorflow as tf

import os
import utils.datasets
import numpy as np

from tinynn.converter import TFLiteConverter


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
    parser.add_argument('--data', type=str, default='./data/8class_5s/QRS_AAMI_8_5s_Re160_10.data ', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='./weights/8class_5s_Re160.pth', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--output', type=str, default='./8class_5s_Re160.onnx', 
                        help='The path where the onnx model is saved')                        
    parser.add_argument('--tfoutput', type=str, default='./8class_5s_Re160', 
                        help='The path where the onnx model is saved')
    parser.add_argument('--GrayScale', type=str, default=True, 
                        help='Input image in Gray Scale: False/True')

# python pytorch2onnx.py --data data/QRS_AAMI_all_10s_320_Se.data --weights modelzoo/MIT_All_10s_Se.pth --output All_10s_SEnet.onnx --tfoutput All_10s_SEnet

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], cfg["backbone"], True, export_onnx = True, imggray = opt.GrayScale).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()

# Torch --> tflite
    torch.quantization.convert(model, inplace=False)
    TFLITE_PATH = opt.tfoutput + "_int8.tflite"
    torch.backends.quantized.engine = 'fbgemm'
  
    if opt.GrayScale:
        test_data = torch.rand(1, 1, cfg["height"], cfg["width"]).to(device)
    else:
        test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    converter = TFLiteConverter(model,
                                test_data,
                                tflite_path=TFLITE_PATH)
    converter.convert()

# Torch --> ONNX --> pb --> tflite
    # test_data = torch.rand(1, 3, cfg["height"], cfg["width"]).to(device)
    # torch.onnx.export(model,                    #model being run
    #                  test_data,                 # model input (or a tuple for multiple inputs)
    #                  opt.output,               # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True)  # whether to execute constant folding for optimization


    # onnx_model = onnx.load(opt.output)  # load onnx model
    # # model_simp, check = simplify(onnx_model)
    # tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    # tf_rep.export_graph(opt.tfoutput)
    # quantize_onnx_model(opt.output, opt.output)

    # TFLITE_PATH = opt.output[0:-5] + ".tflite"
    # converter = tf.lite.TFLiteConverter.from_saved_model(opt.tfoutput)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # converter.representative_dataset = rep_dataset
    # # converter.inference_input_type = tf.int8
    # # converter.inference_output_type = tf.int8
    # tf_lite_model = converter.convert()
    # with open(TFLITE_PATH, 'wb') as f:
    #     f.write(tf_lite_model)

    

