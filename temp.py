from math import floor, sqrt, log, exp
from pickletools import uint8
from re import I
from turtle import color
from unittest import result
from matplotlib import style
import torch
import numpy as np
import cv2
import matplotlib.pylab as plt
import os
import random

if __name__ == '__main__':

    # for i in range(196,396):
    #     ground = cv2.imread('D:/ali/ori/g2.bmp')
    #     img = cv2.imread('D:/ali/timage/x_' + str(i)+ '.bmp') 
    #     if(img is not None):
    #         mask = 0*img
    #         # mask[850:,1280:,:]=1
    #         mask[600:,:1800,:]=1
    #         #mask[:,:,0] = 255
    #         temp = cv2.absdiff(img, ground).sum(2)
    #         img=0*img
    #         img[temp>50,0]=255
    #         img = cv2.medianBlur(img,3)
    #         # img[:,:,0] = get_rect(img[:,:,0])
    #         cv2.imwrite('D:/ali/tlabel/x_' + str(i)+ '.bmp', img*mask)
    #         if(img.sum()<120*255):
    #             os.remove('D:/FFOutput/label/z450m_' + str(i)+ '.png')
            # plt.imshow(sFFOutputencyMap, cmap='gray')#cv2.medianBlur(img,3)cv2.absdiff(img, ground)
            # plt.show()
    # file_list = os.listdir('D:/Download/image/')
    # for file in file_list:
    #     if(file[-1]=='t'):
    #         if(os.path.exists('D:/Download/realonly/'+file[:-4]+'.png')):
    #             if(os.path.exists('D:/Download/realonly/'+file)):
    #                 continue
    #             else:
    #                 os.system("xcopy D:/Download/image/"+file+" D:/Download/realonly/")
            # os.remove('D:/Download/realonly/' + file)
            # os.remove('D:/microlight/image/' + file)
    # with open("D:/microlight/test.txt", "r") as f:
    #     other, plane, person, trunk =0,0,0,0
    #     for line in f.readlines():
            
    #         temp_label = cv2.imread('D:/microlight/label/' + line.strip('\n'))
    #         if(temp_label.sum()==0):
    #             other +=1
    #         elif(line[0]=='t'):
    #             trunk +=1
    #         elif(line[0]=='p' and line[1]=='e'):
    #             person +=1
    #         elif(line[0]=='p' and line[1]=='l'):
    #             plane+=1
    # print(other, plane, person, trunk)
    #生成训练测试txt文件
    # dir = 'D:/Download/col/'
    # file_list = os.listdir(dir)
    # random.shuffle(file_list)
    # bound = int(0.8*len(file_list))
    # for i in range(len(file_list)):
    #     if(file_list[i][-1]=='t'):
    #         with open('D:/Download/coltrain.txt','a') as file:
    #             file.write(file_list[i][:-4]+'.jpg'+'\n')

    # for i in range(bound, len(file_list)):
    #     if(file_list[i][-1]=='g'):
    #         with open('D:/Download/test.txt','a') as file:
    #             file.write(file_list[i]+'\n')

    # file_list = os.listdir('D:/temp/aimage/')
    # for file in file_list:
    #     if(file[-5:]=='_json'):
    #         img = cv2.imread('D:/temp/aimage/' + file +'/label.png')
    #         img[img>10] =255
    #         img[:,:,0]=img[:,:,2]
    #         img[:,:,2]=img[:,:,1]
    #         img[:,:,1]=img[:,:,2]*0
    #         cv2.imwrite('D:/temp/alabel/' + file[:-5] +'.png', img)
    #         ori_img = cv2.imread('D:/temp/aimage/' + file +'/img.png')
    #         cv2.imwrite('D:/temp/aimage/' + file[:-5] +'.png', ori_img)

    #数据增强
    # file_list = os.listdir('D:/Download/tlabel/')
    # for file in file_list:
    #     label = cv2.imread('D:/Download/tlabel/' + file)
    #     image = cv2.imread('D:/Download/timage/' + file)
    #     if(random.random()>0.62):
    #         image = cv2.imread('D:/ali/image/' + file )
    #         label = cv2.imread('D:/ali/label/' + file )
    #         cv2.imwrite('D:/ali/image/' + file , 255-image)
    #     (h, w) = image.shape[:2]
    #     center = (w / 2, h / 2)
            #旋转
            # M = cv2.getRotationMatrix2D(center, angle=5, scale=1.0)
            # cv2.imwrite('D:/ali/image/5' + file , cv2.warpAffine(image, M, (w, h)) ) 
            # cv2.imwrite('D:/ali/label/5' + file , cv2.warpAffine(label, M, (w, h)) ) 
            # N = cv2.getRotationMatrix2D(center, angle=-5, scale=1.0)
            # cv2.imwrite('D:/ali/image/-5' + file , cv2.warpAffine(image, N, (w, h)) ) 
            # cv2.imwrite('D:/ali/label/-5' + file , cv2.warpAffine(label, N, (w, h)) )
            #水平翻转
        # cv2.imwrite('D:/Download/timage/s' + file ,cv2.flip(image, 1))
        # cv2.imwrite('D:/Download/tlabel/s' + file ,cv2.flip(label, 1))
    #         #放缩
    #         M = cv2.getRotationMatrix2D(center, angle=0, scale=0.9)
    #         cv2.imwrite('D:/peitu/image/9' + file , cv2.warpAffine(image, M, (w, h)) ) 
    #         cv2.imwrite('D:/peitu/label/9' + file , cv2.warpAffine(label, M, (w, h)) ) 
    #         N = cv2.getRotationMatrix2D(center, angle=0, scale=1.05)
    #         cv2.imwrite('D:/peitu/image/1' + file , cv2.warpAffine(image, N, (w, h)) ) 
    #         cv2.imwrite('D:/peitu/label/1' + file , cv2.warpAffine(label, N, (w, h)) )

    #加入移动侦测
    # file_list = os.listdir('D:/temp/label/')
    # cap = cv2.VideoCapture('D:/temp/trunk3.mpg')
    # ret, frame = cap.read()
    # while(ret):
    #     before = frame
    #     ret, frame = cap.read()
    #     diff_img = cv2.absdiff(frame, before).astype(np.int16)

    #     for file in file_list:
    #         if(file[:6]=='trunk3'):
    #             img = cv2.imread('D:/temp/image/' + file)
    #             count = cv2.absdiff(frame, img).sum()

    #             if(count<100):             
    #                 print(count)
    #                 frame[:, :, 0] = diff_img[:, :, 0]
    #                 cv2.imwrite('D:/temp/image/' + file, frame)

    #微光视频播放
    # cap = cv2.VideoCapture('D:/Download/yudabao.mp4') 
    # ret, frame = cap.read()
    # print(frame.shape)
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fps = 25
    # out = cv2.VideoWriter('D:/Download/yudabao.avi', fourcc, fps, (960, 536))
    # #before = cv2.imread('D:/microlight/image/ground_3n.png')
    # while(ret): 
    #     cv2.imshow('result', frame)
    #     cv2.waitKey(20)
    #     # diff_img = cv2.absdiff(frame, before)
    #     if(frame[0,0,0]>0):
    #         out.write(frame.astype(np.uint8))
    #     ret, frame = cap.read()
    # cap.release() 
    # cv2.destroyAllWindows()

    ##删除冗余
    # file_list1 = os.listdir('D:/ali/label/')
    # file_list2 = os.listdir('D:/ali/image/')
    # rongyu = list(set(file_list2).difference(set(file_list1)))
    # for file in file_list1:
    #     image = cv2.imread('D:/ali/label/' + file )
    #     if(image.sum()<255*30):
    #         os.remove('D:/ali/image/' + file)
    #         os.remove('D:/ali/label/' + file)
        #os.remove('D:/microlight/image/' + file)
        # image = cv2.imread('D:/temp/aimage/' + file )
        # cv2.imwrite('D:/temp/image/' + file[:-4], image)
        # label = cv2.imread('D:/temp/alabel/' + file )
        # cv2.imwrite('D:/temp/label/' + file[:-4], label)

    ##裁切图片
    # file_list = os.listdir('D:/Download/timage/')
    # for file in file_list:

    #for i in range(125, 261):
        #file =  'people_5n_' + str(i)+ '.png'
    # image = cv2.imread('D:/ali/image/' + file )
    # label = cv2.imread('D:/ali/label/' + file )
    # if(image is not None):
    #     for i in range(3):
    #         for j in range(3):
    #             if(np.sum(label[360*i:360*(i+1), 640*j:640*(j+1), 0])>0):
    #                 cv2.imwrite('D:/ali/image/' + file[:-4] +'_'+ str(3*i+j+1) + '.png', image[360*i:360*(i+1), 640*j:640*(j+1), :])
    #                 cv2.imwrite('D:/ali/label/' + file[:-4] +'_'+ str(3*i+j+1) + '.png', label[360*i:360*(i+1), 640*j:640*(j+1), :])
    # with open("D:/ali/test.txt", "r") as f:
    #     for line in f.readlines():
    # for i in range(150, 451):
        
        # if(file[0]=='h'):
        #     image = cv2.imread('D:/Download/timage/' + file)
        #     label = cv2.imread('D:/Download/tlabel/' + file)
        #     temp = label[:,:,1]*0
        #     temp[label[:,:,2]>0]=255
        #     x,y,w,h = cv2.boundingRect(temp)
        #     print(x,y,w,h)
        #     center_x = floor(x +w/2)
        #     center_y = floor(y +h/2)
        #     up = center_x-320
        #     down = center_x+320
        #     left = center_y-180
        #     right = center_y+180
        #     if(left<0):
        #         left = 0
        #         right = 360
        #     if(right>1080):
        #         left = 720
        #         right = 1080
        #     if(up<0):
        #         up = 0
        #         down = 640
        #     if(down >1920):
        #         up = 1280
        #         down = 1920
        #     cv2.imwrite('D:/Download/tlabel/' + file, label[left:right, up:down, :])
        #     cv2.imwrite('D:/Download/timage/' + file, image[left:right, up:down, :])

    ##处理数据集
    # for i in range(1,172):
    #     filelist = os.listdir('C:/Users/DELL/Downloads/data/'+str(i)+'/')
    #     j=0
    #     for file in filelist:
    #         if(file[-5]=='m'):
    #             label = cv2.imread('C:/Users/DELL/Downloads/data/'+str(i)+'/'+file)
    #             temp = label*0
    #             height, weight, _ = label.shape
    #             for h in range(height):
    #                 for w in range(weight):
    #                     color = tuple(label[h, w])
    #                     if(color == (159,255,96)or color == (0,80,255)):
    #                         temp[h,w,1]=255
    #                     if(color == (255,32,0)or color == (255,191,0)):
    #                         temp[h,w,0]=255
    #                     if(color == (0,0,143)or color == (0,255,255)or color == (0,0,175)):
    #                         temp[h,w,2]=255
    #             cv2.imwrite('D:/spdata/label/' + str(i)+ str(j)+ '.png', temp)
    #             image = cv2.imread('C:/Users/DELL/Downloads/data/'+str(i)+'/'+file[:-6]+'.jpg')
    #             image = cv2.resize(image, (80, 160), interpolation=cv2.INTER_AREA)
    #             cv2.imwrite('D:/spdata/image/' + str(i)+ str(j)+ '.png', image)
    #             j+=1

    ##伪背景数据集
    # filelist = os.listdir('D:/spdata/label/')
    # ground = os.listdir('D:/ali/ground/')
    # for file in filelist:
    #     label = cv2.imread('D:/spdata/label/'+file)
    #     image = cv2.imread('D:/spdata/image/'+file)
    #     g1 = cv2.imread('D:/ali/ground/'+ground[random.randint(0,25)])
    #     g2 = cv2.imread('D:/ali/ground/'+ground[random.randint(0,25)])
    #     #deal with g1
    #     imGray = cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),1)
    #     imGray = cv2.merge([imGray, imGray, imGray])
    #     lbgray = cv2.flip(label, 1)
    #     mask = 0*g1
    #     imtemp = 0*g1
    #     x = random.randint(0,550)
    #     y = random.randint(0,190)
    #     imtemp[y:y+160, x:x+80, :] = imGray
    #     mask[y:y+160, x:x+80, :] =lbgray
    #     imtemp[mask.sum(2)<128,:]=g1[mask.sum(2)<128,:]
    #     cv2.imwrite('D:/ali/label/' + file[:-4]+ 'a.png', mask)
    #     cv2.imwrite('D:/ali/image/' + file[:-4]+ 'a.png', imtemp)
    #     #deal with g2
    #     image = cv2.resize(image, (40, 80), interpolation=cv2.INTER_AREA)
    #     label = cv2.resize(label, (40, 80), interpolation=cv2.INTER_AREA)
    #     mask2 = 0*g2
    #     imtemp2 = 0*g2
    #     x2 = random.randint(0,590)
    #     y2 = random.randint(0,270)
    #     imtemp2[y:y+80, x:x+40, :] = image
    #     mask2[y:y+80, x:x+40, :] = label
    #     imtemp2[mask2.sum(2)<128,:]=g2[mask2.sum(2)<128,:]
    #     cv2.imwrite('D:/ali/label/' + file[:-4]+ 'b.png', mask2)
    #     cv2.imwrite('D:/ali/image/' + file[:-4]+ 'b.png', imtemp2)

    ##分割数据转热点图长宽图
    # file_list = os.listdir('D:/test_lighht/')
    # for file in file_list:
    #     if(file[0]=='t' ):
    #         label = cv2.imread('D:/microlight/label/' + file )
    #         image = cv2.imread('D:/microlight/image/' + file )
    #         temp = label[:,:,0]
    #         temp[temp>0]=255
    #         x,y,w,h = cv2.boundingRect(temp.astype(np.uint8))
    #         center_x = floor(x +w/2)
    #         center_y = floor(y +h/2)
    #         w = floor(w/2)
    #         h = floor(h/2)
    #         dst = np.zeros_like(temp).astype(np.float)
    #         for i in range(-w,w):
    #             for j in range(-h, h):
    #                 dst[center_y+j,center_x+i] = 1.0 - (i/w)*(i/w) - (j/h)*(j/h)
    #         dst[dst<0]=0.0
    #         myresult = np.zeros_like(label)
    #         myresult[:,:,1] = (255*dst).astype(np.uint8)
    #         #myresult[center_y,center_x,1] = w
    #         myresult[center_y,center_x,2] = h
    #         cv2.imwrite('D:/Download/tlabel/' + file, myresult)
    #         cv2.imwrite('D:/Download/timage/' + file, image)
    # file_list = os.listdir('D:/Download/label/')
    # for file in file_list:
    #     label = cv2.imread('D:/Download/label/' + file )
    #     temp = label[:,:,0].astype(np.float32)/255.0
    #     temp = 1.0- np.sqrt(1.0-temp)
    #     label[:,:,0] = (255*temp).astype(np.uint8)
    #     label[:,:,1:3]=0
    #     cv2.imwrite('D:/Download/llabel/' + file, label)

    ##密集行人数据集生成
    # filelist = os.listdir('D:/spdata/label/')
    # for im in range(len(filelist)):
    #     label = cv2.imread('D:/spdata/label/'+filelist[im])
    #     image = cv2.imread('D:/spdata/image/'+filelist[im])
    #     temp = label.sum(2)
    #     temp[temp>0]=255
    #     x,y,w,h = cv2.boundingRect(temp.astype(np.uint8))
    #     center_x = floor(x +w/2)
    #     center_y = floor(y +h/2)
    #     w = floor(w/2)
    #     h = floor(h/2)
    #     dst = np.zeros_like(temp).astype(np.float)
    #     for i in range(-w,w):
    #         for j in range(-h, h):
    #             dst[center_y+j,center_x+i] = 1.0 - (i/w)*(i/w) - (j/h)*(j/h)
    #     dst[dst<0]=0.0
    #     label[:,:,0] = (255*dst).astype(np.uint8)
    #     label[:,:,1:3] =0
    #     image_a = cv2.imread('D:/Download/image/'+str(im//16)+'.png')
    #     label_a = cv2.imread('D:/Download/label/'+str(im//16)+'.png')
    #     xtemp = random.randint(0,20)
    #     col = (im//8)%2
    #     raw = im%8
    #     image_a[160*col+xtemp:160*(col+1)+xtemp, 80*raw:80*(raw+1), :]=image
    #     label_a[160*col+xtemp:160*(col+1)+xtemp, 80*raw:80*(raw+1), :]=label
    #     cv2.imwrite('D:/Download/image/'+str(im//16)+'.png', image_a)
    #     cv2.imwrite('D:/Download/label/'+str(im//16)+'.png', label_a)

    ##热点图txt生成
    # filelist = os.listdir('D:/Download/label/')
    # for file in filelist:
    #     if(os.path.exists('D:/Download/image/'+file[:-4]+'.txt')):
    #         continue
    #     else:
    #         label = cv2.imread('D:/Download/label/'+file)
    #         for k in range(label.shape[2]):
    #             for i in range(label.shape[0]):
    #                 for j in range(label.shape[1]):
    #                     if(label[i,j,k]==255):
    #                         w, h = 0, 0
    #                         while(label[i-w,j,k]>0):  
    #                             w+=1
    #                         while(label[i,j-h,k]>0): 
    #                             h+=1
    #                         with open('D:/Download/image/'+file[:-4]+'.txt','a') as txt:
    #                             txt.write("{:d} {:d} {:d} {:d} {:d}\n".format(k, i, j, 2*w-1, 2*h-1))

    # dir = 'D:/yolov5-master/yolov5-master/runs/detect/exp24/'
    # file_list = os.listdir(dir)
    # for file in file_list:
    #     if(file[-4:]=='.txt' ):#and file[-5]>='0' and file[-5]<='9' and len(file)<=7
    #         img = cv2.imread('D:/source_temp/Snapshot/road/350m-car/'+ file[:-4]+ '.jpg')
    #         cv2.imwrite('D:/Download/delete/' + file[:-4]+ '.png', img)
    #         #if(img is not None and img.shape[0]<1000 ):
    #             #cv2.imwrite('D:/Download/realonly/'+file[:-4]+'.png', img)

    #         with open(dir+file,"r") as f:
    #             for line in f.readlines():
    #                 line = line.strip('\n')
    #                 numbers = list(map(float, line.split())) #转化为浮点数 
    #                 x = int(1920*numbers[1]+0.5)
    #                 y = int(1080*numbers[2]+0.5)
    #                 w = int(1920*numbers[3]+0.5)
    #                 h = int(1080*numbers[4]+0.5)
    #                 #0:person;2:car;7:trunk;4:airplane;5:bus;19:cow
    #                 with open('D:/Download/delete/'+file[:-4]+'.txt','a') as txt:
    #                     if(numbers[0]==0.0):
    #                         txt.write("{:d} {:d} {:d} {:d} {:d}\n".format(0, x, y, w, h))
    #                     if(numbers[0]==2.0 or numbers[0]==5.0 or numbers[0]==7.0 or numbers[0]==19.0):
    #                         txt.write("{:d} {:d} {:d} {:d} {:d}\n".format(1, x, y, w, h))
    #                     if(numbers[0]==4.0):
    #                         txt.write("{:d} {:d} {:d} {:d} {:d}\n".format(2, x, y, w, h))
    # import shutil


    # file_list = os.listdir('D:/coco/row/')
    # for file in file_list:
        #     with open('D:/Download/temp/'+file,'r') as txt:
        #         data =[]
        #         for line in txt.readlines():
        #             line = line.strip('\n')
        #             numbers = list(map(int, line.split())) #转化为数
        #             if (numbers[0]<9):
        #                 data.append("{:d} {:d} {:d} {:d} {:d}\n".format(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]))
        #             else:
        #                 data.append("{:d} {:d} {:d} {:d} {:d}\n".format(numbers[0]-10, (numbers[1]+numbers[3])//2, (numbers[2]+numbers[4])//2, numbers[3]-numbers[1],numbers[4]-numbers[2]))

        #         f = open('D:/Download/temp/'+file, "w")    #以写入的形式打开txt文件
        #         f.writelines(data)          #将修改后的文本内容写入
        #         f.close()  # 关闭文件
            # img = cv2.imread('D:/Download/train/' + file )
            # hist = cv2.calcHist([img], [0], None, [256], [0, 255])
            # plt.plot(hist, color="r")
            # if(img.shape[0]!=360):
            #     print(file[:-4])
            #     new = np.zeros((360,640,3))+127
            #     new[:-1,:,:]=img
            #     cv2.imwrite('D:/Download/train/' + file, new)
            
        #     cv2.imwrite('D:/Download/test/o%03d.png'%i, img)
        #     os.remove('D:/Download/realonly/' + file)
        #     i+=1
            # img = cv2.imread('D:/Download/train/' + file )
            # img = cv2.flip(img, 1)
            # #img[:,:,1] = cv2.equalizeHist(img[:,:,1])
            # noise = np.random.normal(0, 0.07, img.shape)
            # img = (img.astype(np.float16)/255.0 + noise)*255
            # img[img<0]=0
            # cv2.imwrite('D:/Download/train/' + file , img.astype(np.uint8))
            # with open('D:/Download/train/'+file[:-4]+'.txt','r') as txt:
            #     data =[]
            #     for line in txt.readlines():
            #         line = line.strip('\n')
            #         numbers = list(map(int, line.split())) #转化为数
            #         data.append("{:d} {:d} {:d} {:d} {:d}\n".format(numbers[0], 640-numbers[1], numbers[2], numbers[3], numbers[4]))
            #     f = open('D:/Download/test/'+file, "w")    #以写入的形式打开txt文件
            #     f.writelines(data)          #将修改后的文本内容写入
            #     f.close()  # 关闭文件
                    # if (numbers[0]<5):
                    #     img[numbers[2]-numbers[4]//2, numbers[1]-numbers[3]//2:numbers[1]+numbers[3]//2, numbers[0]] = 255
                    #     img[numbers[2]+numbers[4]//2-1, numbers[1]-numbers[3]//2:numbers[1]+numbers[3]//2, numbers[0]] = 255
                    #     img[numbers[2]-numbers[4]//2:numbers[2]+numbers[4]//2, numbers[1]-numbers[3]//2, numbers[0]] = 255
                    #     img[numbers[2]-numbers[4]//2:numbers[2]+numbers[4]//2, numbers[1]+numbers[3]//2-1, numbers[0]] = 255
            # plt.imshow(img.astype(np.uint8))
            # plt.show()


    # file_list = os.listdir('D:/Download/')
    # for file in file_list:
    #     if(file[0]=='u' and file[-1]=='g'):
    #         img = cv2.imread('D:/Download/' + file[:-4]+'.png')
    #         tmp =img[:,:,0]
    #         img[:,:,0] = img[:,:,1]
    #         img[:,:,1] =tmp
    #         cv2.imwrite('D:/Download/pp' + file[-5]+'.png', img)
            
    #         with open('D:/Download/delete/'+file,'r') as txt:
    #             data =[]
    #             for line in txt.readlines():
    #                 line = line.strip('\n')
    #                 numbers = list(map(int, line.split())) #转化为数
    #                 temp[numbers[2], numbers[1]-numbers[3]//2:numbers[1]+numbers[3]//2] = 255
    #                 temp[numbers[2]-numbers[4]//2:numbers[2]+numbers[4]//2, numbers[1]] = 255
    #         x,y,w,h = cv2.boundingRect(temp)
    #         if(w>=600 or h>=320):
    #             img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    #             cv2.imwrite('D:/Download/test/' + file[:-4]+'.png', img)
    #             with open('D:/Download/delete/'+file,'r') as txt:
    #                 data =[]
    #                 for line in txt.readlines():
    #                     line = line.strip('\n')
    #                     numbers = list(map(int, line.split())) #转化为数
    #                     numbers[1:] = [int((a + 1.5)/3) for a in numbers[1:]]
    #                     data.append("{:d} {:d} {:d} {:d} {:d}\n".format(numbers[0], numbers[1], numbers[2], numbers[3], numbers[4]))
    #                 f = open('D:/Download/test/'+file, "w")    #以写入的形式打开txt文件
    #                 f.writelines(data)          #将修改后的文本内容写入
    #                 f.close()  # 关闭文件
    #         else:
    #             center_x = floor(x +w/2)
    #             center_y = floor(y +h/2)
    #             up = center_x-320
    #             down = center_x+320
    #             left = center_y-180
    #             right = center_y+180
    #             if(left<0):
    #                 left = 0
    #                 right = 360
    #             if(right>1080):
    #                 left = 720
    #                 right = 1080
    #             if(up<0):
    #                 up = 0
    #                 down = 640
    #             if(down >1920):
    #                 up = 1280
    #                 down = 1920
    #             cv2.imwrite('D:/Download/test/' + file[:-4]+'.png', img[left:right, up:down, :])
    #             with open('D:/Download/delete/'+file,'r') as txt:
    #                 data =[]
    #                 for line in txt.readlines():
    #                     line = line.strip('\n')
    #                     numbers = list(map(int, line.split())) #转化为数
    #                     data.append("{:d} {:d} {:d} {:d} {:d}\n".format(numbers[0], numbers[1]-up, numbers[2]-left, numbers[3], numbers[4]))
    #                 f = open('D:/Download/test/'+file, "w")    #以写入的形式打开txt文件
    #                 f.writelines(data)          #将修改后的文本内容写入
    #                 f.close()  # 关闭文件
##转移文件
    # file_list = os.listdir('D:/Download/test/')
    # i,j=0,0
    # for file in file_list:
    #     if(file[-1]=='t' and os.path.getsize('D:/Download/test/'+file)>1):
    #         img = cv2.imread('D:/Download/test/' + file[:-4]+'.png')
    #         cv2.imwrite('D:/Download/train/v%03d.png'%i, img)
    #         os.remove('D:/Download/test/' + file[:-4]+'.png')
    #         with open('D:/Download/test/'+file,'r') as txt:
    #             f = open('D:/Download/train/v%03d.txt'%i, "w")    #以写入的形式打开txt文件
    #             f.writelines(txt.readlines())          #将文本内容写入
    #             f.close()  # 关闭文件
    #         os.remove('D:/Download/test/' + file)
    #         i+=1
    #     elif(file[-1]=='t'):
    #         os.remove('D:/Download/test/' + file)
    #         img = cv2.imread('D:/Download/test/' + file[:-4]+'.png')
    #         cv2.imwrite('D:/Download/test/v%03d.png'%j, img)
    #         os.remove('D:/Download/test/' + file[:-4]+'.png')
    #         j+=1
    # dp1=np.zeros(33)
    # dp2=np.zeros(33)
    # dp3=np.zeros(33)
    # dp1[0]=6/33
    # dp2[1]=5*6/33/32
    # dp3[2]=4*5*6/33/32/31
    # tmp=0
    # for i in range(1,28):
    #     dp1[i]=dp1[i-1]*(28-i)/(33-i)
    # for i in range(2,29):
    #     dp2[i]=dp2[i-1]*(29-i)/(33-i)*i/(i-1)
    # for i in range(3,30):
    #     dp3[i]=dp3[i-1]*(30-i)/(33-i)*i/(i-2)
    #plt.plot(np.arange(33),dp1+np.flipud(dp2)+dp3+dp2+np.flipud(dp3)+np.flipud(dp1))
    # plt.plot(np.arange(33),dp3,np.flipud(dp3))
    # plt.plot(np.arange(33),np.flipud(dp2),np.flipud(dp1))
    # plt.bar(np.arange(33)+1, dp1)
    # plt.bar(np.arange(33)+1, dp2,bottom=dp1)
    # plt.bar(np.arange(33)+1, dp3,bottom=dp1+dp2)
    # plt.bar(np.arange(33)+1, np.flipud(dp3),bottom=dp1+dp2+dp3)
    # plt.bar(np.arange(33)+1, np.flipud(dp2),bottom=dp1+dp2+dp3+np.flipud(dp3))
    # plt.bar(np.arange(33)+1, np.flipud(dp1),bottom=dp1+dp2+dp3+np.flipud(dp3)+np.flipud(dp2))
    # plt.show()
    ##绘制曲线图
    x=np.arange(10)*0.05+0.5
    dp1=[0.98249704,0.9767271,0.969533,0.9560757,0.9387902,0.87431014,0.7935765 ,0.67632663,0.31314322,0.05046237]#[12,18,22,30,39,51,67,67,50,43,34,28,23,19,15,13]#
    dp2=[0.9651162,0.9534883,0.9387953 ,0.9126801,0.7945123,0.7104977 ,0.56344175,0.35924545,0.07416259,0.00159437]#[12,15,17,20,24,28,33,33,50,43,34,28,23,19,15,13]#np.zeros(101)
    dp3=[0.9999997,0.9999997,0.9999997,0.9999997,0.9999997,0.94472927,0.89729506,0.63707983,0.46274787,0]#[12,18,22,30,39,51,67,67,50,43,34,28,23,19,15,13]#np.zeros(101)
    dp4=[0.9043807,0.90357345,0.900225,0.899344 ,0.8974864,0.8930248 ,0.8892599,0.8773205,0.86705554,0.796909]#[12,18,22,30,39,51,67,67,50,43,34,28,23,19,15,13]#np.zeros(101)
    dp5=[0.67612743,0.67612743,0.67612743,0.67612743,0.67612743,0.67612743,0.67612743,0.6613805,0.6613805, 0.5715165]#
    dp6=[0.5916374,0.5916374,0.5916374,0.5916374,0.5916374,0.5916374,0.5916374,0.5916374,0.5916374,0.42950565]#
    # for i in range(101):
    #     dp1[i]=1.0 - (abs(i-50)/50)#int(0.375*(dp1[i-2]+dp1[i-1]))
    #     dp2[i]=1.0 - 2*(abs(i-50)/50)#int(0.375*(dp2[i-2]+dp2[i-1]))
    #     dp3[i]=1.0 - sqrt(abs(i-50)/50)#int(0.25*(dp3[i-2]+dp3[i-1]))
    #     dp4[i]=1.0 - ((i-50)/50)*((i-50)/50)#int(0.4*(dp4[i-2]+dp4[i-1]))
    #     dp5[i]=1.0 - 2*((i-50)/50)*((i-50)/50)
    # dp2[dp2<0]=0
    # dp5[dp5<0]=0
    plt.plot(x,dp1,label='人(ours)',color='r')
    plt.plot(x,dp2,label='车(ours)',color='b')
    plt.plot(x,dp3,label='飞机(ours)',color='g')
    plt.plot(x,dp4,linestyle='--',label='人(YOLOV5s)',color='r')
    plt.plot(x,dp5,linestyle='--',label='车(YOLOV5s)',color='b')
    plt.plot(x,dp6,linestyle='--',label='飞机(YOLOV5s)',color='g')
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False
    plt.legend( loc='lower left')  # loc='upper left'
    plt.xlabel('IoU 阈值')
    plt.ylabel('平均正确率')
    plt.title('不同类别平均正确率曲线')
    plt.show()
    ##绘制热点图
    # from DetOH import get_heatmap
    # dst=np.zeros((135,72,3))
    # x=67
    # y=36
    # w=67
    # h=36
    # for i in range(-w,w+1):
    #     for j in range(-h, h+1):
    #         if(x+i>=0 and x+i<dst.shape[0] and y+j>=0 and y+j<dst.shape[1]):
    #             dst[x+i,y+j] = 1.0 - (i/w)*(i/w) - (j/h)*(j/h)#2*abs(i/w) - 2*abs(j/h)#
    #             #dst[x+i,y+j] = 1.0 - 0.5*(i*i/w) - 0.5*(j*j/h)
    # dst[dst<0]=0.0
    # cv2.imwrite("D:/hm.png",(dst*255).astype(np.uint8))
