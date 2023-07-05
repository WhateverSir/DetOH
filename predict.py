from math import ceil, floor, sqrt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import numpy as np
import cv2
import os
import time
import matplotlib.pylab as plt
from model import my_cnn, stage2_cnn, ResNet, Wide_Deep, Fibo_Dense#, Unet
from mycenternet import Unet, double_conv2d_bn, deconv2d_bn
from shuffle_v2 import shuffle_Unet, double_conv2d_bn_sf, deconv2d_bn_sf
from DeepLab import DeepLabV3Plus
from thop import profile
import torch.nn as nn


def test_in(label, result):
    num0 = label[0,:,:].sum()
    temp = label[0,:,:]
    temp[temp>0]=1
    num1 =(temp*result[0,:,:]).sum()
    if(num1>=0.7*num0):
        return 1
    else:
        return 0



    #平移标注
    # def find_move(img1, img2, label):
    #     img1 = img1[:,:,0]
    #     img2 = img2[:,:,0]
    #     sum_a = np.sum(img1 * label)
    #     thresh = 1e8
    #     result = label
    #     for i in range(-3, 3):
    #         for j in range(-3, 3):
    #             newlabel = img_move(label, i, j)
    #             newsum = np.sum(img2 * newlabel)
    #             dist = (sum_a-newsum) * (sum_a-newsum)
    #             if(dist < thresh):
    #                 thresh = dist
    #                 result = newlabel
    #     return result
    # def img_move(label, a, b):
    #     result = 0*label
    #     for i in range(label.shape[0]):
    #         for j in range(label.shape[1]):
    #             if(i+a>=0 and i+a<label.shape[0] and j+b>=0 and j+b<label.shape[1]):
    #                 result[i][j] = label[i+a][j+b]
    #     return result
    # label = cv2.imread('D:/peitu/label/trunk (6)_' + str(214)+ '.png')
    # for i in range(201):
    #     img = cv2.imread('D:/peitu/image/trunk (6)_' + str(214+i)+ '.png')       
    #     #person = label[:,:,2]/255
    #     trunk = label[:,:,-1]/255
    #     img1 = cv2.imread('D:/peitu/image/trunk (6)_' + str(214+i+1)+ '.png')
    #     #newperson = find_move(img, img1, person)
    #     newtrunk = find_move(img, img1, trunk)
    #     #result = cv2.merge([255*newperson.astype(np.uint8), 0*person.astype(np.uint8), 255*person.astype(np.uint8)])
    #     result = cv2.merge([0*trunk.astype(np.uint8), 0*trunk.astype(np.uint8), 255*newtrunk.astype(np.uint8)])
    #     label = result
    #     cv2.imwrite('D:/peitu/label/trunk (6)_' + str(214+i+1)+ '.png', result.astype(np.uint8))

    ##后处理
def _nms(heat, kernel=11):
	#'通过maxpooling的方式找出3*3邻域内最大的点'
	conv1 = nn.Conv2d(1, 1, (9, 9), stride=1, padding=4)
	conv1.weight = nn.Parameter(torch.ones((1, 1, 9, 9)))
	temp_heat = conv1(heat.to(torch.float32).unsqueeze(1)).squeeze(0)
	pad = (kernel - 1) // 2
	hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
	keep = (hmax == heat).float()#'判断最大的点是否是当前点'
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
    thresh = 0.5
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
        while(ys[i]-j>=0 and heat[ys[i]-j, xs[i]]>thresh):
            j+=1
        k=0
        while(ys[i]+k<heat.shape[0] and heat[ys[i]+k, xs[i]]>thresh):
            k+=1
        w[i] = int(k*k/(1-heat[ys[i]+k, xs[i]])+j*j/(1-heat[ys[i]-j, xs[i]])+0.5)//2
        j=0
        while(xs[i]-j>=0 and heat[ys[i], xs[i]-j]>thresh):
            j+=1
        k=0
        while(xs[i]+k<heat.shape[1] and heat[ys[i], xs[i]+k]>thresh):
            k+=1
        h[i] = int(k*k/(1-heat[ys[i], xs[i]+k])+j*j/(1-heat[ys[i], xs[i]-j])+0.5)//2
    return w,h


if __name__ == '__main__':
    device = 'cuda:0'
    model = torch.load('D:/Download/mobunet3_0224_fl.pth')
    model.eval()
    #print(model.out.bias.data)
    #算力计算
    input = torch.randn(1, 3, 640, 360).to('cuda') #模型输入的形状,batch_size=1
    flops, params = profile(model, inputs=(input, ))
    print("Flops(G):",flops/1e9, "Parameters(M):",params/1e6) #flops单位G，para单位M
    ##图片测试
    read_path = 'D:/Download/train/'
    write_path = 'D:/Download/result/test10/'
    with open('D:/Download/test.txt','r') as f:
        for line in f.readlines():
            line = line.strip('\n')

            img = cv2.imread(read_path+line)
            img_t = torch.tensor(np.transpose(img.astype(np.float32)/255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
            result =np.transpose(model(img_t).squeeze().detach().cpu().numpy() , (1, 2, 0))
            # #result=(255*result).astype(np.uint8)
            # plt.imshow((255*result).astype(np.uint8))
            # plt.show()
            with open(write_path+line[:-4]+'.txt','a') as wf:
                for catid in range(3):

                    scores, ys, xs = ctdet_decode(torch.tensor(result[:,:,catid]).unsqueeze(0), result.astype(np.uint8))#
                    w, h = cal_wh2(result[:,:,catid], scores, ys, xs)
                    #result=(255*result).astype(np.uint8)
                    for i in range(len(ys)):
                        if(scores[i]>0.35):
                            # img[int(ys[i]-w[i]):int(ys[i]+w[i]),int(xs[i]),catid]=255
                            # img[int(ys[i]),int(xs[i]-h[i]):int(xs[i]+h[i]),catid]=255
                            wf.write("%s %s %s %s %s %s\n" % (catid, ys[i], xs[i], w[i], h[i], scores[i]))

            # plt.imshow(np.vstack([img, (255*result).astype(np.uint8)]))
            # plt.show()
            # b, g, r = cv2.split(img)
            # cv2.imwrite('D:/Download/' + file, cv2.merge((r, g, b)))

    
    ##视频测试
    # cap = cv2.VideoCapture('D:/Download/yudabao.avi') 
    # ret, frame = cap.read()
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fps = 13
    # out = cv2.VideoWriter('D:/microlight/temp.avi', fourcc, fps, (1920, 1080))D:/microlight/plane.avi
    # out.write(frame)
    # while(ret): 
    #     before = frame
    #     ret, frame = cap.read()
        # mean0 = np.expand_dims(frame.mean(0), 0)
        # mean1 = np.expand_dims(frame.mean(1), 1)
        # ground = 0.5*np.abs(0*frame+mean0)+0.5*np.abs(0*frame+mean1)
        # frame = 1.5*frame.astype(np.float16)- 0.5*ground
        # frame = frame.astype(np.float32)
        # frame[:,:,0]= 2*np.abs(frame[:,:,0]-mean0)
        # frame[:,:,1]= 2*np.abs(frame[:,:,1]-mean1)
        # frame[:,:,2]= np.where(frame[:,:,0] > frame[:,:,1], frame[:,:,0], frame[:,:,1])
        # diffimg = cv2.resize(frame, (120, 67),interpolation=cv2.INTER_LINEAR)
        # diffimg = cv2.absdiff(frame, cv2.resize(diffimg, (960, 536), interpolation=cv2.INTER_AREA))
        # img_t = torch.tensor(np.transpose(frame.astype(np.float32)/255.0, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(device)
        # result = model(img_t).squeeze().detach().cpu().numpy()#np.transpose(, (1, 2, 0))
        # #frame+=(80*result).astype(np.uint8)
        # scores, ys, xs = ctdet_decode(torch.tensor(result).unsqueeze(0), result.astype(np.uint8))
        # w, h = cal_wh2(result, scores, ys, xs)
        # frame[:,:,1]=(255*result).astype(np.uint8)
        # for i in range(len(ys)):
        #     if(scores[i]>0.65):
        #         frame[ys[i]-w[i]:ys[i]+w[i], xs[i], 1] = 255
        #         frame[ys[i], xs[i]-h[i]:xs[i]+h[i], 1] = 255
                # frame[ys[i]-w[i]:ys[i]+w[i], xs[i]-h[i], 1] = 255
                # frame[ys[i]-w[i]:ys[i]+w[i], xs[i]+h[i], 1] = 255
                # frame[ys[i]-w[i], xs[i]-h[i]:xs[i]+h[i], 1] = 255
                # frame[ys[i]+w[i], xs[i]-h[i]:xs[i]+h[i], 1] = 255
        # cv2.imshow('result', frame.astype(np.uint8))
        # cv2.waitKey(20)
        #out.write(frame.astype(np.uint8))
    # cap.release() 
    # cv2.destroyAllWindows()
