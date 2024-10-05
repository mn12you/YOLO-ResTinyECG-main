import os
import torch
import argparse
from tqdm import tqdm


from torchsummary import summary

import utils.utils
import utils.datasets
import model.detector

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='',
                        help='The path of the model')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

     # Gray scale img
    if cfg["backbone"].find('Sh')>0 or cfg["backbone"].find('csp')>0: imggray = False 
    else: imggray = True 
    convGray = 3
    if imggray:
        convGray = 1
   


    assert os.path.exists(opt.weights), "请指定正确的模型路径"

    #打印消息
    print("评估配置:")
    print("model_name:%s"%cfg["model_name"])
    print("width:%d height:%d"%(cfg["width"], cfg["height"]))
    print("val:%s"%(cfg["val"]))
    print("model_path:%s"%(opt.weights))
    
    #加载数据
    val_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False, imggray = imggray)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=False,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )
    
    #指定后端设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #初始化模型
    load_param = False
    print(imggray)
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], cfg["backbone"], load_param = False, imggray = imggray, quantize = False).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))
    #sets the module in eval node
    model.eval()

    #打印模型结构
    # summary(model, input_size=(1, cfg["height"], cfg["width"]))
    
    #模型评估
    conf_thres=0.1
    print("computer mAP...")
    precision, recall, AP, f1, p, r, ap = utils.utils.evaluation(val_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
    # print("computer PR...")
    # precision, recall, _, f1, p, r, _ = utils.utils.evaluation(val_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
    print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))
    print("Precision of each class:")
    print(p)
    print("Recall of each class:")
    print(r)
    print("AP of each class:")
    print(ap)
