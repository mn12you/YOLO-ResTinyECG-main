import os
import cv2
import time
import argparse
import numpy as np
import torch
import model.detector
import utils.utils
from tqdm import tqdm
import utils.datasets
from utils.loss import compute_loss

def cam_show_img(img, feature_map, grads, out_name):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		
    grads = grads.reshape([grads.shape[0],-1])					
    weights = np.mean(grads, axis=1)							
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    cv2.imwrite(out_name, cam_img)

if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/8class_5s/QRS_AAMI_8_5s_Re160_S05_1.data ', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default='./weights/QRS_8class_5s_Re160-1.pth', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='data/AAMI_5s_320/signal_resize/test_fold1.txt', 
                        help='The path of test image')
    parser.add_argument('--GrayScale', type=str, default=True, 
                        help='Input image in Gray Scale: False/True')
    parser.add_argument('--visualize', type=str, default=False, 
                        help='Visualize features: False/True')

# python test.py --data data/QRS_AAMI_10s_320.data --weights modelzoo/QRS_AAMI_10s_320.pth --img ./data/AAMI_10s_320_w/signal_resize/test_fold1.txt
#  python test.py --data data/QRS_AAMI_all_5s_320.data --weights modelzoo/QRS_AAMI_all_5s_320.pth --img ./data/AAMI_5s_320_w/signal_resize/test_fold1.txt
# python test.py --data data/QRS_AAMI_8_5s_Re_80.data --weights weights/QRS_8class_5s_Se_80_1-59-epoch-0.823708ap-model.pth --img D:/Yolo-fastest-backup/data/AAMI_5s_320_w/signal_resize/test_fold1.txt

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    assert os.path.exists(opt.weights), "请指定正确的模型路径"
    assert os.path.exists(opt.img), "请指定正确的测试图像路径"
    # save_path = opt.img
    
    # test_dataset = utils.datasets.TensorDataset(opt.img, cfg["width"], cfg["height"], imgaug = False)
    # #測試集
    # test_dataloader = torch.utils.data.DataLoader(test_dataset,
    #                                              batch_size=1,
    #                                              shuffle=False,
    #                                              collate_fn=utils.datasets.collate_fn,
    #                                              num_workers=4,
    #                                              pin_memory=False,
    #                                              drop_last=False,
    #                                              persistent_workers=True
    #                                              )

    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], cfg["backbone"], True, export_onnx = False, imggray = opt.GrayScale).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    #sets the module in eval node
    model.eval()
    
    visualize = opt.visualize
    if visualize:
    # require grad
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers        

    with open(opt.img,'r') as imgf:
        imgp = imgf.read().splitlines()        
        for save_path in tqdm(imgp):  
            save_path = save_path.replace('\\', '/')    
            #数据预处理
            ori_img = cv2.imread(save_path)        
            res_img = cv2.resize(ori_img, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
            if opt.GrayScale:
                res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)  # convert to gray scale
                img = res_img.reshape(1, cfg["height"], cfg["width"], 1)
            else:
                img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
            img = torch.from_numpy(img.transpose(0,3, 1, 2))
            img = img.to(device).float() / 255.0            

            #模型推理
            # start = time.perf_counter()
            preds = model(img)
            # end = time.perf_counter()
            # time = (end - start) * 1000.
            # print("forward time:%fms"%time)

            # if visualize:
            #     # grad-cam
            #     preds = model(img)
                
            #     model.zero_grad()
            #     targets = torch.zeros(2, 6)
            #     lbox, lobj, lcls, loss = compute_loss(preds, targets.to(device), cfg, device)
            #     loss.requires_grad_(True)
            #     loss.backward()

            #     _grads = model.grads_list
            #     _grads.reverse()
            #     _features = model.features_list

            #     # for g, f in zip(_grads, _features):
            #     #     print('grad', type(g), g.shape)
            #     #     print('feature', type(f), f.shape)
                
            #     for i in [17, 20, 23]:
            #         out_name = save_path[0:save_path.index('/s')] + '/CAM' + save_path[save_path.rindex('/s'):len(save_path)]
            #         cam_show_img(img, _features[i].cpu().detach().numpy()[0], _grads[i].cpu().detach().numpy()[0], out_name)

            #     preds = preds[0]
            # else:
            #     preds = model(img)

            # # print('preds\n')
            # # print([len(a) for a in preds])
            # preds_np = tuple(t.cpu() for t in preds[0][0])   
            # preds_nparray = [t.detach().numpy() for t in preds_np]
            # # print(preds_nparray)  
            # preds_nparray = np.transpose(preds_nparray)
            # with open('Re80_preds_T.csv', 'w') as f:          
            #     for idx in range(len(preds_nparray)):
            #         # f.write([preds_nparray[idx][idx2] for idx2 in range(len(preds_nparray[idx]))])
            #         np.savetxt(f, [preds_nparray[idx][idx2] for idx2 in range(len(preds_nparray[idx]))], delimiter=",", fmt='% f')
            #         np.savetxt(f, [idx], delimiter=",", fmt='% d')
                       
            # #特征图后处理
            output = utils.utils.handel_preds(preds, cfg, device)
            # # # print('output\n')
            # # # print(output)
            # # output_np = np.array(output[0])              
            # # np.savetxt("Re80_outputs.csv", [output_np[idx] for idx in range(len(output_np))], delimiter=",", fmt='% f')            
            output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.1, iou_thres = 0.5)
            # # # print('output_boxes\n')
            # # # print(output_boxes)                      
            # # # torch tensor 转为 numpy array
            # # output_boxes_np = np.array(output_boxes[0])        
            # # # print(len(output_boxes_np[0]))     
            # # np.savetxt("Re80_output_boxes.csv", [output_boxes_np[idx] for idx in range(len(output_boxes_np))], delimiter=",", fmt='% f')

            # break
    
            #加载label names
            LABEL_NAMES = []
            with open(cfg["names"], 'r') as f:
                for line in f.readlines():
                    LABEL_NAMES.append(line.strip())
            
            h, w, _ = ori_img.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]  
            # scale_h = 1 
            # scale_w = 1     

            #绘制预测框
            for box in output_boxes[0]:
                box = box.tolist()
            
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                if int(box[5]) == 0:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (255, 0, 0), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (255, 0, 0), 1)
                elif int(box[5]) == 1:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (0, 255, 0), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (0, 255, 0), 1)
                elif int(box[5]) == 2:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (0, 0, 255), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (0, 0, 255), 1)
                elif int(box[5]) == 3:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (100, 200, 0), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (100, 200, 0), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (100, 200, 0), 1)
                elif int(box[5]) == 4:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 200, 100), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (0, 200, 100), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (0, 200, 100), 1)
                elif int(box[5]) == 5:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (255, 0, 255), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (255, 0, 255), 1)
                else:
                    cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.4, (0, 0, 0), 1)	
                    cv2.putText(ori_img, category, (x1, y1 - 15), 0, 0.5, (0, 0, 0), 1)
              
            # save_path = save_path[0:save_path.index('/s')] + '/box' + save_path[save_path.rindex('/s'):len(save_path)] 
            # print(save_path)
            # cv2.imwrite(save_path, ori_img)
    

