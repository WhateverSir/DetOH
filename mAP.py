from collections import Counter

import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pylab as plt

from DetOH import Unet, deconv2d_bn, double_conv2d_bn
from predict import ctdet_decode, cal_wh2, cal_wh


def mean_average_precision(pred_bboxes,true_boxes,iou_threshold,num_classes=3):
    
    #pred_bboxes(list): [[train_idx,class_pred,prob_score,x1,y1,x2,y2], ...]
    
    average_precisions=[]#存储每一个类别的AP
    epsilon=1e-6#防止分母为0
    
    #对于每一个类别
    for c in range(num_classes):
        detections=[]#存储预测为该类别的bbox
        ground_truths=[]#存储本身就是该类别的bbox(GT)
        
        for detection in pred_bboxes:
            if detection[1]==c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1]==c:
                ground_truths.append(true_box)
                
        #img 0 has 3 bboxes
        #img 1 has 5 bboxes
        #就像这样：amount_bboxes={0:3,1:5}
        #统计每一张图片中真实框的个数,train_idx指示了图片的编号以区分每张图片
        amount_bboxes=Counter(gt[0] for gt in ground_truths)
        
        for key,val in amount_bboxes.items():
            amount_bboxes[key]=torch.zeros(val)#置0，表示这些真实框初始时都没有与任何预测框匹配
        #此时，amount_bboxes={0:torch.tensor([0,0,0]),1:torch.tensor([0,0,0,0,0])}
        
        #将预测框按照置信度从大到小排序
        detections.sort(key=lambda x:x[2],reverse=True)
        
        #初始化TP,FP
        TP=torch.zeros(len(detections))
        FP=torch.zeros(len(detections))
        
        #TP+FN就是当前类别GT框的总数，是固定的
        total_true_bboxes=len(ground_truths)
        
        #如果当前类别一个GT框都没有，那么直接跳过即可
        if total_true_bboxes == 0:
            continue
        
        #对于每个预测框，先找到它所在图片中的所有真实框，然后计算预测框与每一个真实框之间的IoU，大于IoU阈值且该真实框没有与其他预测框匹配，则置该预测框的预测结果为TP，否则为FP
        for detection_idx,detection in enumerate(detections):
            #在计算IoU时，只能是同一张图片内的框做，不同图片之间不能做
            #图片的编号存在第0个维度
            #于是下面这句代码的作用是：找到当前预测框detection所在图片中的所有真实框，用于计算IoU
            ground_truth_img=[bbox for bbox in ground_truths if bbox[0]==detection[0]]
            
            num_gts=len(ground_truth_img)
            
            best_iou=0
            for idx,gt in enumerate(ground_truth_img):
                #计算当前预测框detection与它所在图片内的每一个真实框的IoU
                iou=insert_over_union(torch.tensor(detection[3:]),torch.tensor(gt[2:]))
                if iou >best_iou:
                    best_iou=iou
                    best_gt_idx=idx
            if best_iou>iou_threshold:
                #这里的detection[0]是amount_bboxes的一个key，表示图片的编号，best_gt_idx是该key对应的value中真实框的下标
                if amount_bboxes[detection[0]][best_gt_idx]==0:#只有没被占用的真实框才能用，0表示未被占用（占用：该真实框与某预测框匹配【两者IoU大于设定的IoU阈值】）
                    TP[detection_idx]=1#该预测框为TP
                    amount_bboxes[detection[0]][best_gt_idx]=1#将该真实框标记为已经用过了，不能再用于其他预测框。因为一个预测框最多只能对应一个真实框（最多：IoU小于IoU阈值时，预测框没有对应的真实框)
                else:
                    FP[detection_idx]=1#虽然该预测框与真实框中的一个框之间的IoU大于IoU阈值，但是这个真实框已经与其他预测框匹配，因此该预测框为FP
            else:
                FP[detection_idx]=1#该预测框与真实框中的每一个框之间的IoU都小于IoU阈值，因此该预测框直接为FP
                
        TP_cumsum=torch.cumsum(TP,dim=0)
        FP_cumsum=torch.cumsum(FP,dim=0)
        
        #套公式
        recalls=TP_cumsum/(total_true_bboxes+epsilon)
        precisions=torch.divide(TP_cumsum,(TP_cumsum+FP_cumsum+epsilon))
        
        #把[0,1]这个点加入其中
        precisions=torch.cat((torch.tensor([1]),precisions))
        recalls=torch.cat((torch.tensor([0]),recalls))
        #使用trapz计算AP
        average_precisions.append(torch.trapz(precisions,recalls))
        
    return average_precisions#sum(average_precisions)/len(average_precisions) 
 
 
def insert_over_union(boxes_preds,boxes_labels):
    
    box1_x1=boxes_preds[...,0:1]
    box1_y1=boxes_preds[...,1:2]
    box1_x2=boxes_preds[...,2:3]
    box1_y2=boxes_preds[...,3:4]#shape:[N,1]
    
    box2_x1=boxes_labels[...,0:1]
    box2_y1=boxes_labels[...,1:2]
    box2_x2=boxes_labels[...,2:3]
    box2_y2=boxes_labels[...,3:4]
    
    x1=torch.max(box1_x1,box2_x1)
    y1=torch.max(box1_y1,box2_y1)
    x2=torch.min(box1_x2,box2_x2)
    y2=torch.min(box1_y2,box2_y2)
    
    
    #计算交集区域面积
    intersection=(x2-x1).clamp(0)*(y2-y1).clamp(0)
    
    box1_area=abs((box1_x2-box1_x1)*(box1_y1-box1_y2))
    box2_area=abs((box2_x2-box2_x1)*(box2_y1-box2_y2))
    
    return intersection/(box1_area+box2_area-intersection+1e-6)

# def nms(pred_boxes, item):
#     flag = True
#     for i in range(len(pred_boxes)):
#         if(pred_boxes[i][0]==item[0] and pred_boxes[i][1]==item[1]):
#             box1_area = (pred_boxes[i][5]-pred_boxes[i][3])*(pred_boxes[i][6]-pred_boxes[i][4])
#             box2_area = (item[5]-item[3])*(item[6]-item[4])
#             if(pred_boxes[i][5]<)
#             iou = intersection/(box1_area+box2_area-intersection+1e-6)
#             if(iou>0.5):
#                 flag = False
#     if(flag):
#         pred_boxes.append(item)
#     return pred_boxes

def get_heatmap(numbers, img):
    dst = np.zeros_like(img[:,:,0]).astype(np.float)
    if(len(numbers)<5):
        return dst
    x = numbers[2]
    y = numbers[1]
    w = numbers[4]//2
    h = numbers[3]//2
    for i in range(-w,w+1):
        for j in range(-h, h+1):
            if(x+i>=0 and x+i<dst.shape[0] and y+j>=0 and y+j<dst.shape[1]):
                dst[x+i,y+j] = 1.0 - abs(i/w) - abs(j/h)#(i/w)*(i/w) - (j/h)*(j/h)#
    dst[dst<0]=0.0
    return dst

if __name__ == '__main__':
    ##图片测试
    prob=0.9
    pred_boxes, true_boxes = [],[]
    with open('D:/Download/test.txt','r') as f:
        id = 0
        for filename in f.readlines():
            id += 1
            filename = filename.strip('\n')

            with open('D:/Download/train/'+filename[:-4]+'.txt','r') as tf:
                 for line in tf.readlines():
                    line = line.strip('\n')
                    numbers = list(map(int, line.split())) #转化为数
                    # heat = get_heatmap(numbers, img)          
                    # temp_label = np.where(heat > temp_label, heat, temp_label)
                    x1 = numbers[1] - numbers[3]//2
                    y1 = numbers[2] - numbers[4]//2
                    x2 = numbers[1] + numbers[3]//2
                    y2 = numbers[2] + numbers[4]//2
                    true_boxes.append([id, numbers[0], x1,y1,x2,y2])

            #自己模型使用
            with open('D:/Download/result/test10/'+filename[:-4]+'.txt','r') as pf:
                 for line in pf.readlines():
                    line = line.strip('\n')
                    numbers = list(map(float, line.split())) #转化为数
                    x1 = numbers[1] - numbers[3]
                    y1 = numbers[2] - numbers[4]
                    x2 = numbers[1] + numbers[3]
                    y2 = numbers[2] + numbers[4]
                    pred_boxes.append([id, int(numbers[0]),numbers[5], int(y1),int(x1),int(y2),int(x2)])
            #YOLOV5使用
            # if os.path.exists('D:/yolov5-master/yolov5-master/runs/detect/train/'+filename[:-4]+'.txt'):
            #     with open('D:/yolov5-master/yolov5-master/runs/detect/train/'+filename[:-4]+'.txt','r') as pf:
            #         for line in pf.readlines():
            #             line = line.strip('\n')
            #             numbers = list(map(float, line.split())) #转化为数
            #             x = int(640*numbers[1]+0.5)
            #             y = int(360*numbers[2]+0.5)
            #             w = int(640*numbers[3]+0.5)
            #             h = int(360*numbers[4]+0.5)
            #             if(numbers[0]==0.0):
            #                 pred_boxes.append([id, 0,numbers[5], x-w/2,y-h/2,x+w/2,y+h/2])
            #             if(numbers[0]==2.0 or numbers[0]==5.0 or numbers[0]==7.0 or numbers[0]==19.0):
            #                 pred_boxes.append([id, 1,numbers[5], x-w/2,y-h/2,x+w/2,y+h/2])
            #             if(numbers[0]==4.0):
            #                 pred_boxes.append([id, 2,numbers[5], x-w/2,y-h/2,x+w/2,y+h/2])
    #print(pred_boxes[:10],true_boxes[:10])
    AP=[]
    for iou_t in range(10):
        iou_threshold = 0.5+iou_t*0.05
        print("AP%s :\n"% (iou_threshold))
        tmp = mean_average_precision(pred_boxes,true_boxes,iou_threshold=iou_threshold,num_classes=3)
        AP.append(np.array(tmp))
        print(np.array(tmp))
    AP=np.array(AP)
    print(np.mean(AP),"AP",np.mean(AP,axis=0))
    print(np.mean(AP,axis=1),"AP 50-95")
    
