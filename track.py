from math import ceil, floor, sqrt
from turtle import color
import torch
import numpy as np
import cv2
import os
import time
import matplotlib.pylab as plt
from model import my_cnn, stage2_cnn, ResNet, Wide_Deep, Fibo_Dense#, Unet
from mycenternet import Unet, double_conv2d_bn, deconv2d_bn
from DeepLab import DeepLabV3Plus
from thop import profile
import torch.nn as nn


    ##后处理
def _nms(heat, kernel=19):
	#'通过maxpooling的方式找出3*3邻域内最大的点'
	conv1 = nn.Conv2d(1, 1, (9, 9), stride=1, padding=4)
	conv1.weight = nn.Parameter(torch.ones((1, 1, 9, 9)))
	temp_heat = conv1(heat.unsqueeze(0)).squeeze(0)
	pad = (kernel - 1) // 2
	hmax = nn.functional.max_pool2d(temp_heat, (kernel, kernel), stride=1, padding=pad)
	keep = (hmax == temp_heat).float()#'判断最大的点是否是当前点'
	return heat * keep
def _topk(scores, K=40):
	cat, height, width = scores.size()
	#'torch.topk,scores[N,C,H*W],返回最大的K个值和其所在的位置'
	topk_scores, topk_inds = torch.topk(scores.reshape(cat,-1), K)
	#'每个N每个C上最大的K个点的坐标，大小分别是[N,C,K]，存储纵坐标和横坐标'
	topk_ys   = (topk_inds / width).int()
	topk_xs   = (topk_inds % width).int()
	return topk_scores, topk_ys, topk_xs
 
def ctdet_decode(heat, wh, K=40):
    cat, height, width = heat.size()
	# perform nms on heatmaps
    heat = _nms(heat)#'判断当前点是否为邻域内最大点，若是则保留，否则位0'
	#'返回最大k个关键点所对应的置信度、一维表示的坐标、通道id也就是类别、横纵坐标，其中inds是从map中索引wh、reg等真值的序号 '
    scores, ys, xs = _topk(heat, K=K)
    scores = scores.squeeze().numpy() 
    ys = ys.squeeze().numpy() 
    xs = xs.squeeze().numpy()
    
    return scores, ys, xs

def cal_wh(heat, scores, ys, xs, K=40):
    w = ys*0
    h = xs*0
    for i in range(K):
        ans = heat[ys[i]-7:ys[i]+8, xs[i]].sum()
        w[i] = ceil(sqrt(280/(15.0*scores[i]-ans)))
        ans_h = heat[ys[i], xs[i]-7:xs[i]+8].sum()
        h[i] = ceil(sqrt(280/(15.0*scores[i]-ans_h)))
    return w,h
def cal_wh2(heat, scores, ys, xs, K=40):
    w = ys*0
    h = xs*0
    thresh = 0.75
    for i in range(K):
        j=0
        while(ys[i]-j>=0 and heat[ys[i]-j, xs[i]]>thresh):
            j+=1
        k=0
        while(ys[i]+k<heat.shape[0] and heat[ys[i]+k, xs[i]]>thresh):
            k+=1
        w[i] = int(0.5*(j + k)/(1.0-thresh)+0.5)
        j=0
        while(xs[i]-j>=0 and heat[ys[i], xs[i]-j]>thresh):
            j+=1
        k=0
        while(xs[i]+k<heat.shape[1] and heat[ys[i], xs[i]+k]>thresh):
            k+=1
        h[i] = int(0.5*(j + k)/(1.0-thresh)+0.5)
    return w,h
def cal_wh3(heat, scores, ys, xs, K=40):
    w = ys*0
    h = xs*0
    thresh = 0.5
    for i in range(K):
        j=0
        while(ys[i]-j>=0 and heat[ys[i]-j, xs[i]]>thresh*scores[i]):
            j+=1
        k=0
        while(ys[i]+k<heat.shape[0] and heat[ys[i]+k, xs[i]]>thresh*scores[i]):
            k+=1
        w[i] = int(k*k/(1-heat[ys[i]+k, xs[i]])+j*j/(1-heat[ys[i]-j, xs[i]])+0.5)//2
        j=0
        while(xs[i]-j>=0 and heat[ys[i], xs[i]-j]>thresh*scores[i]):
            j+=1
        k=0
        while(xs[i]+k<heat.shape[1] and heat[ys[i], xs[i]+k]>thresh*scores[i]):
            k+=1
        h[i] = int(k*k/(1-heat[ys[i], xs[i]+k])+j*j/(1-heat[ys[i], xs[i]-j])+0.5)//2
    return w,h

def kalman(xk, zk, pk, r=0.1):
    gk = pk/(pk+r)
    xk1 = xk + gk*(zk - xk)
    pk = (1-gk)*pk
    return xk1, pk
def insert_over_union(x,y,w,h,x_,y_,w_,h_):
    
    box1_x1=x - w
    box1_y1=y - h
    box1_x2=x + w
    box1_y2=y + h
    
    box2_x1=x_ - w_
    box2_y1=y_ - h_
    box2_x2=x_ + w_
    box2_y2=y_ + h_
    
    in_h = min(box1_y2,box2_y2) - max(box1_y1,box2_y1)
    in_w = min(box1_x2,box2_x2) - max(box1_x1,box2_x1)
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    #计算交集区域面积
    
    box1_area=4*w*h
    box2_area=4*w_*h_
    
    return inter/(box1_area+box2_area-inter+1e-6)
if __name__ == '__main__':
    device = 'cuda:0'
    model = torch.load('D:/Download/unet_0428.pth')
    model.eval()
    colors = [[0,0,0],[255,255,255],[255,0,255],[0,255,255],[255,255,0],[0,0,255],[255,0,0],[0,255,0],[255,100,255],[100,255,255],[255,255,100],[128,0,128],[255,128,0],[0,128,255],[128,255,0],[128,0,255],[255,0,128],[0,255,128]]
    ##视频测试
    cap = cv2.VideoCapture('D:/Download/yudabao.avi') 
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    ##第一帧进行各个初始化
    img_t = torch.tensor(np.transpose(frame.astype(np.float32)/255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
    result = model(img_t).squeeze().detach().cpu().numpy()
    scores, ys, xs = ctdet_decode(torch.tensor(result).unsqueeze(0), result.astype(np.uint8))
    w, h = cal_wh2(result, scores, ys, xs)
    list = []#目标列表
    id = 0
    for i in range(len(ys)):
        temp_np=np.zeros((9))
        if(scores[i]>0.618):
            temp_np[0]=id#id
            id+=1
            temp_np[1]=scores[i]#置信度
            temp_np[2]=ys[i]#横坐标
            temp_np[3]=xs[i]#纵坐标
            temp_np[4]=w[i]#宽
            temp_np[5]=h[i]#高
            temp_np[6]=0.#横向速度分量
            temp_np[7]=0.#纵向速度分量
            temp_np[8]=0#丢失帧数
            list.append(temp_np)
    temp_frame = np.pad(frame,((10,10),(10,10),(0,0)),'constant', constant_values=127)
    for item in list:
        item_ = item.astype(np.int)
        item_[2]+=10
        item_[3]+=10
        temp_frame[item_[2]-item_[4]:item_[2]+item_[4], item_[3]-item_[5], :] = colors[item_[0]%11]
        temp_frame[item_[2]-item_[4]:item_[2]+item_[4], item_[3]+item_[5], :] = colors[item_[0]%11]
        temp_frame[item_[2]-item_[4], item_[3]-item_[5]:item_[3]+item_[5], :] = colors[item_[0]%11]
        temp_frame[item_[2]+item_[4], item_[3]-item_[5]:item_[3]+item_[5], :] = colors[item_[0]%11]
    print('%03d th list:\n'%id, np.array(list).astype(np.int))
    
    plt.imshow( temp_frame.astype(np.uint8))
    plt.show()
    ##后续帧进行跟踪匹配
    while(ret): 
        before = frame
        ret, frame = cap.read()

        #已有目标匹配
        for item in list:
            temp_y, temp_x, thresh = 0,0,1e5
            num = item.astype(np.int)
            obj = before[num[2]-num[4]:num[2]+num[4], num[3]-num[5]:num[3]+num[5], :]
            for i in range(-3, 3):
                for j in range(-7, 7):
                    temp_frame = np.pad(frame,((10,10),(10,10),(0,0)),'constant', constant_values=127)
                    newobj = temp_frame[num[2]-num[4]+i+10+num[6]:num[2]+num[4]+i+10+num[6], num[3]-num[5]+j+10+num[7]:num[3]+num[5]+j+10+num[7], :]
                    
                    dist = cv2.absdiff(obj, newobj)
                    if(dist.sum() < thresh):
                        thresh = dist.sum()
                        temp_y, temp_x = i, j
            if(thresh<1e5):
                item[2]+=temp_y+item[6]
                item[3]+=temp_x+item[7]
                item[6]=item[6]+temp_y
                item[7]=item[7]+temp_x
                item[8]+=1
            else:
                item[2]+=item[6]
                item[3]+=item[7]
                item[8]+=1
        #目标检测模型调用
        img_t = torch.tensor(np.transpose(frame.astype(np.float32)/255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
        result = model(img_t).squeeze().detach().cpu().numpy()
        scores, ys, xs = ctdet_decode(torch.tensor(result).unsqueeze(0), result.astype(np.uint8))
        w, h = cal_wh2(result, scores, ys, xs)
        #新目标加入
        ##思路：新目标加入与匹配，对列表目标也进行更新。
        for i in range(len(ys)):
            temp_np=np.zeros((9))
            max_iou = 0.09
            for item in list:
                iou = insert_over_union(item[2],item[3],item[4],item[5],ys[i],xs[i],w[i],h[i])
                if(max_iou<iou):
                    max_iou = iou
                if(iou>0.25):
                    item[6]=0.6*item[6]+0.4*(ys[i]-item[2])
                    item[7]=0.6*item[7]+0.4*(xs[i]-item[3])
                    item[2]=0.6*item[2]+0.4*ys[i]
                    item[3]=0.6*item[3]+0.4*xs[i]
                    item[4]=0.6*item[4]+0.4*w[i]#宽
                    item[5]=0.6*item[5]+0.4*h[i]#高
                    item[8]=0
            if(max_iou < 0.1 and scores[i]>0.8 and w[i]*h[i]>36):
                temp_np[0]=id#id
                id+=1
                temp_np[1]=scores[i]#置信度
                temp_np[2]=ys[i]#横坐标
                temp_np[3]=xs[i]#纵坐标
                temp_np[4]=w[i]#宽
                temp_np[5]=h[i]#高
                temp_np[6]=0.#横向速度分量
                temp_np[7]=0.#纵向速度分量
                temp_np[8]=0#丢失帧数
                list.append(temp_np)
        #丢失目标删除
        for i in reversed(range(len(list))):
            if(list[i][8]>9 or list[i][4]<5):
                list.pop(i)
            if(list[i][2]-list[i][4]+list[i][6]<0 or list[i][3]-list[i][5]+list[i][7]<0 or list[i][2]+list[i][4]>frame.shape[0] or list[i][3]+list[i][5]+list[i][7]>frame.shape[1]):
                list.pop(i)
        temp_frame = np.pad(frame,((10,10),(10,10),(0,0)),'constant', constant_values=127)
        for item in list:
            item_ = item.astype(np.int)
            item_[2]+=10
            item_[3]+=10
            temp_frame[item_[2]-item_[4]:item_[2]+item_[4], item_[3]-item_[5], :] = colors[item_[0]%18]
            temp_frame[item_[2]-item_[4]:item_[2]+item_[4], item_[3]+item_[5], :] = colors[item_[0]%18]
            temp_frame[item_[2]-item_[4], item_[3]-item_[5]:item_[3]+item_[5], :] = colors[item_[0]%18]
            temp_frame[item_[2]+item_[4], item_[3]-item_[5]:item_[3]+item_[5], :] = colors[item_[0]%18]
        print('%03d th list:\n'%id, np.array(list).astype(np.int))
        
        cv2.imshow('track',temp_frame.astype(np.uint8))
        #plt.show()
        cv2.waitKey(20)
    cap.release() 
    cv2.destroyAllWindows()
