import cv2
import time

import torch
import torchvision
import torch.nn.functional as F

import os, time
import numpy as np
from tqdm import tqdm

#加载data
def load_datafile(data_path):
    #需要配置的超参数
    cfg = {"model_name":None,
    
           "epochs": None,
           "steps": None,           
           "batch_size": None,
           "subdivisions":None,
           "learning_rate": None,
           "ClassW": None,

           "backbone": None,
           "pre_weights": None,        
           "classes": None,
           "width": None,
           "height": None,           
           "anchor_num": None,
           "anchors": None,

           "all": None,
           "val": None,           
           "train": None,
           "names":None
        }

    assert os.path.exists(data_path), "请指定正确配置.data文件路径"

    #指定配置项的类型
    list_type_key = ["anchors", "steps"]
    str_type_key = ["model_name", "all", "val", "train", "names", "pre_weights", "ClassW", "backbone"]
    int_type_key = ["epochs", "batch_size", "classes", "width",
                   "height", "anchor_num", "subdivisions"]
    float_type_key = ["learning_rate"]
    
    #加载配置文件
    with open(data_path, 'r') as f:
        for line in f.readlines():
            if line == '\n' or line[0] == "[":
                continue
            else:
                data = line.strip().split("=")
                #配置项类型转换
                if data[0] in cfg:
                    if data[0] in int_type_key:
                       cfg[data[0]] = int(data[1])
                    elif data[0] in str_type_key:
                        cfg[data[0]] = data[1]
                    elif data[0] in float_type_key:
                        cfg[data[0]] = float(data[1])
                    elif data[0] in list_type_key:
                        cfg[data[0]] = [float(x) for x in data[1].split(",")]
                    else:
                        print("配置文件有错误的配置项")
                else:
                    print("%s配置文件里有无效配置项:%s"%(data_path, data))
    return cfg

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return np.mean(p), np.mean(r), np.mean(ap), np.mean(f1), p, r, ap

def get_batch_statistics(outputs, targets, iou_threshold, device, width, wh_ratio):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    #labels = []
    min_wh = width * wh_ratio
    max_wh = width - min_wh
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        #print(target_labels)
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                                
                pred_box = pred_box.to(device)                
                pred_label = pred_label.to(device)

                ifpred_box = pred_box.unsqueeze(0)
                #print(pred_i)
                #print(ifpred_box[:,0])
                # If box edge too small --> break
                if ifpred_box[0,0] < min_wh:
                    continue
                # If box edge too large --> break
                if ifpred_box[0,0] > max_wh:
                    continue

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label.to(device) not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    #labels[pred_i] = target_labels[0,pred_i]
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.1, classes=None, width=320, wh_ratio=0.01):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    print(conf_thres)
    # print(width)
    # print(wh_ratio)

    nc = prediction.shape[2] - 5  # number of classes

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh = width * wh_ratio
    max_wh = width - min_wh  # 4096 #width     
    max_det = int(round(0.2/wh_ratio))  # maximum number of detections per image --> 0.1/wh_ratio
    # print(max_det)
    max_nms =  max_det * 50  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 1.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [torch.zeros((0, 6), device="cpu")] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints

        # print(x[..., 0:4])
        # print(x[..., 0])
        # print(x[..., 4].max(0))

        # x[((x[..., 0] < min_wh) | (x[..., 0] > max_wh)), 4] = 0  # width-height
        x = x[x[..., 4] > conf_thres]  # confidence
        
        # print(x[:, :4])
        # print(x[:, 5:]) # each cls_conf
        # print(x[:, 4:5]) # obj_conf

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # print(x)
        
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        # c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores, off_class = x[:, :4], x[:, 4], x[:, 5]
        # print(off_class)
        i = torchvision.ops.batched_nms(boxes, scores, off_class, iou_thres)  # NMS with class
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i].detach().cpu()

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def make_grid(h, w, cfg, device):
    hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
    return torch.stack((wv, hv), 2).repeat(1,1,3).reshape(h, w, cfg["anchor_num"], -1).to(device)

#特征图后处理
def handel_preds(preds, cfg, device):
    #加载anchor配置
    anchors = np.array(cfg["anchors"])
    anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, cfg["anchor_num"], 2)).to(device)

    output_bboxes = []
    layer_index = [0, 0, 0, 1, 1, 1]

    for i in range(len(preds) // 3):
        bacth_bboxes = []
        reg_preds = preds[i * 3]
        obj_preds = preds[(i * 3) + 1]
        cls_preds = preds[(i * 3) + 2]

        for r, o, c in zip(reg_preds, obj_preds, cls_preds):
            r = r.permute(1, 2, 0)
            r = r.reshape(r.shape[0], r.shape[1], cfg["anchor_num"], -1)

            o = o.permute(1, 2, 0)
            o = o.reshape(o.shape[0], o.shape[1], cfg["anchor_num"], -1)

            c = c.permute(1, 2, 0)
            c = c.reshape(c.shape[0],c.shape[1], 1, c.shape[2])
            c = c.repeat(1, 1, 3, 1)

            anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            #计算anchor box的cx, cy
            grid = make_grid(r.shape[0], r.shape[1], cfg, device)
            stride = cfg["height"] /  r.shape[0]
            anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

            #计算anchor box的w, h
            anchors_cfg = anchors[i]
            anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg # wh

            #计算obj分数
            anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

            #计算cls分数
            anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim = 3)

            #torch tensor 转为 numpy array
            anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            bacth_bboxes.append(anchor_boxes)     

        #n, anchor num, h, w, box => n, (anchor num*h*w), box
        bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
        bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1]) 

        output_bboxes.append(bacth_bboxes)    
        
    #merge
    output = torch.cat(output_bboxes, 1)
            
    return output

#模型评估
def evaluation(val_dataloader, cfg, model, device, conf_thres, nms_thresh = 0.4, iou_thres = 0.5):

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    pbar = tqdm(val_dataloader)

    for imgs, targets in pbar:
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)       

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= torch.tensor([cfg["width"], cfg["height"], cfg["width"], cfg["height"]]).to(device)

        #对预测的anchorbox进行nms处理
        with torch.no_grad():
            preds = model(imgs)

            #特征图后处理:生成anchorbox
            output = handel_preds(preds, cfg, device)
            if cfg["all"].find('_5s')>0: wh_ratio=0.01 #0.01
            elif cfg["all"].find('_10s')>0: wh_ratio=0.004
            elif cfg["all"].find('_15s')>0: wh_ratio=0.003
            elif cfg["all"].find('_20s')>0: wh_ratio=0.002
            
            output_boxes = non_max_suppression(output, conf_thres=conf_thres, iou_thres=nms_thresh, width=cfg["width"], wh_ratio=wh_ratio)

        sample_metrics += get_batch_statistics(output_boxes, targets, iou_thres, device, cfg["width"], wh_ratio)
        pbar.set_description("Evaluation model:") 

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    return metrics_output     