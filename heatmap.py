from math import floor, sqrt
import torch
import numpy as np
import cv2
import matplotlib.pylab as plt
import os
import random

def get_heatmap(numbers, img):
    dst = np.zeros_like(img[:,:,0]).astype(np.float)
    x = int(1920*numbers[1])
    y = int(1080*numbers[2])
    w = int(1920*numbers[3]/2)
    h = int(1080*numbers[4]/2)
    # gauss = cv2.getGaussianKernel(2*h+1, 1) * cv2.getGaussianKernel(2*w+1,1).T #高斯核
    # print(gauss.shape)
    # dst[y-h:y+h+1, x-w:x+w+1] = gauss/gauss[h,w]
    for i in range(-w,w):
        for j in range(-h, h):
            dst[y+j,x+i] = 1.0 - (i/w)*(i/w) - (j/h)*(j/h)
    dst[dst<0]=0.0
    return dst

def get_w(numbers, img):
    dst = np.zeros_like(img[:,:,0]).astype(np.float)
    x = int(1920*numbers[1])
    y = int(1080*numbers[2])
    w = 1920*numbers[3]/2
    dst[y,x] = w/255.0
    return dst

def get_h(numbers, img):
    dst = np.zeros_like(img[:,:,0]).astype(np.float)
    x = int(1920*numbers[1])
    y = int(1080*numbers[2])
    h = 1080*numbers[4]/2
    dst[y,x] = h/255.0
    return dst

if __name__ == '__main__':
    dir = 'D:/coco/labels/'#'D:/yolov5-master/yolov5-master/runs/detect/exp7/'
    file_list = os.listdir(dir)
    for file in file_list:
        if(file[0]=='0'):
            img = cv2.imread('D:/coco/val2017/' + file[:-4]+ '.jpg')
            height, width, _ = img.shape
            if(height>width):
                with open(dir+file,"r") as f:
                    data=[]
                    for line in f.readlines():
                        line = line.strip('\n')
                        numbers = list(map(float, line.split())) #转化为浮点数
                        x = int(360*numbers[1]+0.5)
                        y = int(640*numbers[2]+0.5)
                        w = int(360*numbers[3]+0.5)
                        h = int(640*numbers[4]+0.5) 
                        # 0:person;2:car;7:trunk;4:airplane;5:bus;19:cow;20:elephant
                        if(numbers[0]==0.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(0, x, y, w, h))
                        elif(numbers[0]==2.0 or numbers[0]==7.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(1, x, y, w, h))
                        elif(numbers[0]==4.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(2, x, y, w, h))
                        txt = open('D:/coco/col/'+file, "w")    #以写入的形式打开txt文件
                        txt.writelines(data)          #将修改后的文本内容写入
                        txt.close()  # 关闭文件
            else:
                with open(dir+file,"r") as f:
                    data=[]
                    for line in f.readlines():
                        line = line.strip('\n')
                        numbers = list(map(float, line.split())) #转化为浮点数
                        x = int(640*numbers[1]+0.5)
                        y = int(360*numbers[2]+0.5)
                        w = int(640*numbers[3]+0.5)
                        h = int(360*numbers[4]+0.5) 
                        # 0:person;2:car;7:trunk;4:airplane;5:bus;19:cow;20:elephant
                        if(numbers[0]==0.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(0, x, y, w, h))
                        elif(numbers[0]==2.0 or numbers[0]==7.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(1, x, y, w, h))
                        elif(numbers[0]==4.0):
                            data.append("{:d} {:d} {:d} {:d} {:d}\n".format(2, x, y, w, h))
                        txt = open('D:/coco/row/'+file, "w")    #以写入的形式打开txt文件
                        txt.writelines(data)          #将修改后的文本内容写入
                        txt.close()  # 关闭文件
            
            # with open(dir+file,"r") as f:
            #     for line in f.readlines():
            #         line = line.strip('\n')
            #         numbers = list(map(float, line.split())) #转化为浮点数 
            #         #0:person;2:car;7:trunk;4:airplane;5:bus;19:cow;20:elephant
            #         if(numbers[0]==0.0 or numbers[0]==20.0):
            #             heat = get_heatmap(numbers, img)
            #             img[:,:,0] = heat + img[:,:,0]-img[:,:,0]*heat
            #         if(numbers[0]==2.0 or numbers[0]==7.0):
            #             heat = get_heatmap(numbers, img)
            #             img[:,:,1] = heat + img[:,:,1]-img[:,:,1]*heat
            #             img[:,:,2] += get_w(numbers, img)
            #             #img[:,:,2] += get_h(numbers, img)
            # cv2.imwrite('D:/Download/tlabel/' + file[:-4] +'.png', (255*img).astype(np.uint8))
            # plt.imshow(img.astype(np.uint8))
            # plt.show()C = numpy.where(A > B, A, B)
