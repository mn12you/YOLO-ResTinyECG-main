import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm
from numpy.testing._private.utils import print_assert_equal

import torch
from torch import optim
from torch.utils.data import dataset
from numpy.core.fromnumeric import shape

from torchsummary import summary

import utils.loss
import utils.utils
import utils.datasets
import model.detector

# python train.py --data ./data/QRS_AAMI_8_20s_Re_160.data

if __name__ == '__main__':
    # 指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/8class_5s/QRS_AAMI_8_5s_Se_1.data',
                        help='Specify training profile *.data')
    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)

    print("训练配置:")
    print(cfg)

    # Gray scale img
    if cfg["backbone"].find('Sh')>0 or cfg["backbone"].find('csp')>0: imggray = False 
    else: imggray = True 
    convGray = 3
    if imggray:
        convGray = 1

    if cfg["ClassW"].find('NA')>0: weights="" 
    else:
        ## Class weight adjustment
        if cfg["names"].find('all.names')>0: # class: N A V L R a J S F e j E paced f other
            classN = [65000, 2200, 6100, 7000, 6200, 150, 100, 2, 700, 20, 200, 100, 6000, 1000, 1500] 
        
        elif cfg["names"].find('8class.names')>0: # class: N A V L R F paced f other
            # classN = [65000, 2200, 6100, 7000, 6200, 700, 6000, 900, 500]
            if cfg["all"].find('_5s')>0: classN = [65000, 2200, 6100, 7000, 6200, 700, 6000, 900, 500] 
            elif cfg["all"].find('_10s')>0: classN = [70000, 2500, 6500, 7500, 6500, 750, 6500, 900, 600] 
            elif cfg["all"].find('_15s')>0: classN = [70000, 2500, 6500, 7500, 6500, 750, 6500, 900, 600] #[71000, 2500, 7000, 7500, 7000, 800, 6500, 1000, 600] 
            elif cfg["all"].find('_20s')>0: classN = [72000, 2500, 7000, 8000, 7000, 800, 7000, 1000, 600]   

        elif cfg["names"].find('AAMI.names')>0: # class: N A V L R other    
            if cfg["all"].find('_5s')>0: classN = [65000, 2200, 6100, 7000, 6200, 10000] 
            elif cfg["all"].find('_10s')>0: classN = [70000, 2300, 6600, 7500, 6700, 10000] 
            elif cfg["all"].find('_15s')>0: classN = [71000, 2400, 6800, 7700, 6900, 10000] 
            elif cfg["all"].find('_20s')>0: classN = [72000, 2400, 6800, 7800, 7000, 10000] 
        #
        WW = 1    
        if cfg["ClassW"].find('log')>0:
            classN = np.log(classN)   # np.sqrt(classN) / np.log(classN)
            if cfg["ClassW"].find('2')>0: WW = 2                 
        elif cfg["ClassW"].find('sqrt')>0:
            classN = np.sqrt(classN)   # np.sqrt(classN) / np.log(classN)
            if cfg["ClassW"].find('2')>0: WW = 2   
        elif cfg["ClassW"].find('sqrt/log')>0:
            classN = np.sqrt(classN) / np.log(classN)
        weights = (np.max(classN)/WW) * np.ones(len(classN)) / classN
        if cfg["ClassW"].find('norm')>0: weights = weights /np.sum(weights)
       

    # 数据集加载

    train_set = utils.datasets.TensorDataset(cfg["train"], cfg["width"], cfg["height"], imgaug = False, imggray = imggray)
    # Random split
    train_set_size = int(len(train_set) * 0.9)
    valid_set_size = len(train_set) - train_set_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])
    test_dataset = utils.datasets.TensorDataset(cfg["val"], cfg["width"], cfg["height"], imgaug = False, imggray = imggray)

    batch_size = int(cfg["batch_size"] / cfg["subdivisions"])
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 训练集
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=utils.datasets.collate_fn,
                                                   num_workers=nw,
                                                   pin_memory=False,
                                                   drop_last=True,
                                                   persistent_workers=True
                                                   )
    #验证集
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=nw,
                                                 pin_memory=False,
                                                 drop_last=True,
                                                 persistent_workers=True
                                                 )
    #測試集
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 collate_fn=utils.datasets.collate_fn,
                                                 num_workers=4,
                                                 pin_memory=False,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # Count class numbers

    # 指定后端设备CUDA&CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)
    # 判断是否加载预训练模型
    load_param = False
    premodel_path = cfg["pre_weights"]
    if premodel_path != None and os.path.exists(premodel_path):
        load_param = True

    # 初始化模型结构
    model = model.detector.Detector(6, cfg["anchor_num"], cfg["backbone"], load_param, imggray = imggray, quantize = False).to(device)
    # summary(model, input_size=(convGray, cfg["height"], cfg["width"]))


 # 加载预训练模型参数
    if load_param == True:
        model.load_state_dict(torch.load(premodel_path, map_location=device), strict = False)
        print(f"Loaded fine-tuned model parameters: {premodel_path}")

        # Modify the final classification layer to output 5 classes instead of 6
        model.output_cls_layers = torch.nn.Conv2d(model.output_cls_layers.in_channels, 5, 1, 1, 0, bias=True).to(device)
    else:
        print("Initialize weights: model/backbone/backbone.pth")


    # 构建SGD优化器
    optimizer = optim.SGD(params=model.parameters(),
                          lr=cfg["learning_rate"],
                          momentum=0.949,
                          weight_decay=0.0005,
                          )
    # optimizer_state = optimizer.state_dict()
    # optimizer.load_state_dict(optimizer_state)
    # 学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=cfg["steps"],
                                            gamma=0.1)
    # scheduler_state = scheduler.state_dict()
    # scheduler.load_state_dict(scheduler_state)
    
    print('Starting training for %g epochs...' % cfg["epochs"])

    savefilename = './weights/' + cfg["model_name"] +  '_results.csv'
    foldN = cfg["val"] 
    foldN = foldN[cfg["val"].find('fold'):-4]
    with open(savefilename, 'a') as f:
        f.write("%s" % (foldN))
        f.write("\n")
        f.write("Img size, %s, %d" % (cfg["height"], convGray))
        f.write("\n")
        f.write("Class weight, %s" % (cfg["ClassW"] ))
        f.write("\n")
        f.writelines(','.join(str(weights) for weights in weights))
        f.write("\n")
        f.write("Epoch, Precision, Recall, mAP, F1")                
        f.write("\n")

    conf_thres=0.1
    batch_num = 0    
    Loop_break = 0
    total_loss_1 = 100
    for epoch in range(cfg["epochs"]):
        model.train()
        pbar = tqdm(train_dataloader)

        for imgs, targets in pbar:
            # 数据预处理
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            # 模型推理
            preds = model(imgs)
            # loss计算            
            iou_loss, obj_loss, cls_loss, total_loss = utils.loss.compute_loss(preds, targets, cfg, device, weights)
            
            # 反向传播求解梯度
            total_loss.backward()

            #学习率预热
            for g in optimizer.param_groups:
                warmup_num =  5 * len(train_dataloader)
                if batch_num <= warmup_num:
                    scale = math.pow(batch_num/warmup_num, 4)
                    g['lr'] = cfg["learning_rate"] * scale
                    
                lr = g["lr"]

            # 更新模型参数
            if batch_num % cfg["subdivisions"] == 0:
                optimizer.step()
                optimizer.zero_grad()

            # 打印相关信息
            info = "Epoch:%d LR:%f CIou:%f Obj:%f Cls:%f Total:%f" % (
                    epoch, lr, iou_loss, obj_loss, cls_loss, total_loss)
            pbar.set_description(info)
            
            batch_num += 1
        
        if epoch % 10 == 9 and epoch > 0:
            model.eval()
            #模型评估
            print("computer mAP...")
            _, _, AP, _, _, _, _ = utils.utils.evaluation(val_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
            print("computer PR...")
            precision, recall, _, f1, _, _, _ = utils.utils.evaluation(val_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
            print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))          
            
            with open(savefilename, 'a') as f:
                f.write("%d, %f, %f, %f, %f" % (epoch, precision, recall, AP, f1))                
                f.write("\n")

            if total_loss>total_loss_1 or (cfg["epochs"]-epoch)<15: 
                Loop_break += 1
                # 学习率调整
                for g in optimizer.param_groups:
                    g['lr'] = cfg["learning_rate"] * 0.1
            total_loss_1 = total_loss
        
        if Loop_break>1: break
        # else: 
        #     # print("Original learning scheduler...")
        #     scheduler.step() 
        
    # 模型保存
    torch.save(model.state_dict(), "weights/%s-%d-epoch-%fap-model.pth" % (cfg["model_name"], epoch, AP))
    #模型评估  
    model.eval()    
    print("computer mAP...")
    _, _, AP, _, _, _, ap = utils.utils.evaluation(test_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
    print("computer PR...")
    precision, recall, _, f1, p, r, _ = utils.utils.evaluation(test_dataloader, cfg, model, device, conf_thres=conf_thres, nms_thresh=0.4, iou_thres=0.5)
    print("Precision:%f Recall:%f AP:%f F1:%f"%(precision, recall, AP, f1))
    print("Precision of each class:")
    print(p)
    print("Recall of each class:")
    print(r)
    print("AP of each class:")
    print(ap)
    with open(savefilename, 'a') as f:
        f.write("Test, %f, %f, %f, %f" % (precision, recall, AP, f1))                
        f.write("\n")
        f.write("Precision,\n")  
        np.savetxt(f, p, delimiter=",", fmt='% f')              
        f.write("\n")
        f.write("Recall,\n")         
        np.savetxt(f, r, delimiter=",", fmt='% f')       
        f.write("\n")
        f.write("AP,\n")         
        np.savetxt(f, ap, delimiter=",", fmt='% f')       
        f.write("\n")
    
    # torch.save(model,'weights/%s_yolofastest.h5' % (cfg["model_name"]))	python train.py --data ./data/QRS_AAMI_8_5s_Se.data