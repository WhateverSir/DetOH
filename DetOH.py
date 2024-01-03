import argparse
import cv2
import time
import datetime
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pylab as plt
from shuffle_v2 import shuffle_Unet, double_conv2d_bn_sf, deconv2d_bn_sf

class double_conv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,strides=1,padding=1):
        super(double_conv2d_bn,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,
                               kernel_size=kernel_size,
                              stride = strides,padding=padding,bias=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,padding=padding,bias=False)
        # self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,padding=padding,bias=False)
        # self.conv4 = nn.Conv2d(out_channels,out_channels,kernel_size = kernel_size,padding=padding,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.bn3 = nn.BatchNorm2d(out_channels)
        # self.bn4 = nn.BatchNorm2d(out_channels)
    
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        # out = F.relu(self.bn3(self.conv3(out)))
        # out = F.relu(self.bn4(self.conv4(out)))
        return out
# conv_dw为深度可分离卷积
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        # 3x3卷积提取特征，步长为2
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),

        # 1x1卷积，步长为1
        nn.Conv2d(inp, oup, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )    
class deconv2d_bn(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,strides=2):
        super(deconv2d_bn,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,
                                        kernel_size = kernel_size,
                                       stride = strides,bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.layer1_conv = double_conv2d_bn(3,16,strides=2)
        self.layer2_conv = nn.Sequential(
            # 第一个深度可分离卷积，步长为1
            conv_dw(16, 32, 1),  # 208,208,16 -> 208,208,32

            # 两个深度可分离卷积块
            conv_dw(32, 32, 1),  # 208,208,32 -> 104,104,64
            conv_dw(32, 32, 1),

            # 104,104,64 -> 52,52,192
            conv_dw(32, 32, 1),
            conv_dw(32, 32, 1),
        )
        self.layer3_conv = nn.Sequential(
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
            conv_dw(64, 64, 1),
        )
        self.layer4_conv = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
        )
        self.layer5_conv = double_conv2d_bn(128,256)
        self.layer6_conv = double_conv2d_bn(256,128)
        self.layer7_conv = double_conv2d_bn(128,64)
        self.layer8_conv = double_conv2d_bn(64,32)
        self.layer9_conv = double_conv2d_bn(32,16)
        
        self.deconv1 = deconv2d_bn_sf(256,128,kernel_size=5, strides=5)
        self.deconv2 = deconv2d_bn_sf(128,64)
        self.deconv3 = deconv2d_bn_sf(64,32)
        self.deconv4 = deconv2d_bn_sf(32,16)
        self.out = nn.Conv2d(16, 3, kernel_size=1, padding=0,bias=True) 
        
    def forward(self,x):
        conv1 = self.layer1_conv(x)
        
        conv2 = self.layer2_conv(conv1)
        
        conv3 = self.layer3_conv(conv2)
        
        conv4 = self.layer4_conv(conv3)
        pool4 = F.max_pool2d(conv4,5)
        conv5 = self.layer5_conv(pool4)

        convt1 = self.deconv1(conv5)
        concat1 = torch.cat([convt1,conv4],dim=1)
        conv6 = self.layer6_conv(concat1)

        convt2 = self.deconv2(conv6)
        concat2 = torch.cat([convt2,conv3],dim=1)
        conv7 = self.layer7_conv(concat2)
        
        convt3 = self.deconv3(conv7)
        concat3 = torch.cat([convt3,conv2],dim=1)
        conv8 = self.layer8_conv(concat3)
        
        convt4 = self.deconv4(conv8)
        concat4 = torch.cat([convt4,F.interpolate(conv1,  scale_factor=2)],dim=1)
        conv9 = self.layer9_conv(concat4)
        out = self.out(conv9)
        return F.sigmoid(out)

        
def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--workers', '-j', type=int, default=0,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--loaddir', default='D:/Download/mobunet3_0224_ce.pth',
                        help='Directory for loading checkpoint models')
    # training hyper params
    parser.add_argument('--batchsize', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.02)')
    # cuda setting
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device of training')
    # checkpoint and log
    parser.add_argument('--savedir', default='D:/Download/mobunet3_0224_fl.pth',
                        help='Directory for saving checkpoint models')
    args = parser.parse_args()
    return args

def get_heatmap(numbers, img):
    dst = np.zeros_like(img[:,:,0]).astype(np.float)
    if(len(numbers)<5):
        return dst
    x = numbers[2]
    y = numbers[1]
    w = max(1,numbers[4]//2)
    h = max(1,numbers[3]//2)
    for i in range(-w,w+1):
        for j in range(-h, h+1):
            if(x+i>=0 and x+i<dst.shape[0] and y+j>=0 and y+j<dst.shape[1]):
                dst[x+i,y+j] = 1.0 - abs(i/w) - abs(j/h)#2*(i/w)*(i/w) - 2*(j/h)*(j/h)#
                #dst[x+i,y+j] = 1.0 - 0.25*(i*i/w) - 0.25*(j*j/h)
    dst[dst<0]=0.0
    return dst

class dataset(data.Dataset):
    def __init__(self, filedir, expand=True):
        self.images = []
        self.target = []
        file_list = os.listdir(filedir)
        for file in file_list:
            if(file[-1]=='g'):
                temp_image = cv2.imread(filedir + file.strip('\n')).astype(np.float16)/255.0
                self.images.append(np.transpose(temp_image, (2, 0, 1)))
                with open(filedir + file.strip('\n')[:-4]+'.txt',"r") as f:
                    temp_label = np.zeros_like(temp_image).astype(np.float16)
                    for line in f.readlines():
                        line = line.strip('\n') 
                        numbers = list(map(int, line.split())) 
                        #if(numbers[0]==0):  
                        heat = get_heatmap(numbers, temp_image)          
                        temp_label[:,:,numbers[0]] = np.where(heat > temp_label[:,:,numbers[0]], heat, temp_label[:,:,numbers[0]])#
                self.target.append(np.transpose(temp_label, (2, 0, 1)))

                ##数据增强与扩充
                #高斯噪声
                if(expand and random.random()>0.8):
                    noise = np.random.normal(0, 0.07, temp_image.shape) 
                    self.images.append(np.transpose(np.fliplr(temp_image+noise).copy(), (2, 0, 1)))
                    self.target.append(np.transpose(np.fliplr(temp_label).copy(), (2, 0, 1)))
                    # plt.imshow(np.vstack([np.flip(temp_image.copy()+noise, 1),np.flip(temp_label.copy(), 1)]))
                    # plt.show()                    
                #直方图均衡
                #水平翻转
                #反相


    def __getitem__(self, index):
        img = self.images[index]
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.BCELoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()     
class WHLoss(nn.Module):

    def __init__(self, kernel=11):
        super(WHLoss, self).__init__()
        self.kernel = kernel
        self.pad = (kernel - 1) // 2
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, output, target):
        hmax = nn.functional.max_pool2d(output, (self.kernel, self.kernel), stride=1, padding=self.pad)
        mask = (hmax == output).float()
        loss = self.criterion(output*mask, target*mask)/mask.sum()
        return loss
class DiceLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, predictive, target):
        intersection =  torch.sum(predictive * target)
        union = torch.sum(predictive) + torch.sum(target) + self.eps#
        loss = 1 - intersection / union
        return loss

class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device(args.device)

        # dataset and dataloader
        test_dataset= dataset('D:/Download/train/',False)
        train_dataset = dataset('D:/Download/train/')
        #col_dataset = dataset('D:/Download/col/')
        self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=args.batchsize, drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)
        self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batchsize, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
        self.train_num = train_dataset.__len__()
        self.test_num = test_dataset.__len__()
        print("Train images:", self.train_num, "Test images:", self.test_num)
        # create network
        self.model = model.to(self.device)

        # create criterion
        self.criterion = FocalLoss()#nn.BCELoss(reduction='mean')#DiceLoss()#nn.MSELoss(reduction='mean')#
        self.whloss = WHLoss()


    def train(self):
        start_time = time.time()
        max_IOU = 0.004
        for epoch in range(args.epochs):
            self.model.train()
            optimizer =  torch.optim.AdamW(model.parameters(), lr=self.args.lr/(epoch//5+1), weight_decay=5e-5)#torch.optim.SGD(model.parameters(), lr=self.args.lr/5, weight_decay=5e-4)#
            ##混合精度训练
            #model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            epoch_loss, miou, test_miou = 0.0, 0.0, 0.0 
            dataloader =[self.test_loader, self.train_loader]
            for iteration, (images, targets) in enumerate(dataloader[epoch//5%2]):
                iteration = iteration + 1

                images =  torch.as_tensor(images, dtype=torch.float32).to(self.device)
                targets = torch.as_tensor(targets, dtype=torch.float32).to(self.device)#.unsqueeze(1)
                # mask = targets[:,1,:,:]*0
                # mask[targets[:,1,:,:]>0]=1

                outputs = self.model(images)
                loss =81*self.criterion(outputs, targets) + self.whloss(outputs, targets)#self.criterion(outputs, targets)# 
                epoch_loss += loss
                miou += mIOU(outputs[:,0,:,:], targets[:,0,:,:])
                
                optimizer.zero_grad()
                loss.backward()
                ##混合精度训练
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
		        #     scaled_loss.backward()
                optimizer.step()

            with torch.no_grad():
                for _, (images, targets) in enumerate(self.test_loader):
                    images =  torch.as_tensor(images, dtype=torch.float32).to(self.device)
                    targets = torch.as_tensor(targets, dtype=torch.float32).to(self.device)#.unsqueeze(1)
                    outputs = self.model(images)
                    test_miou += mIOU(outputs[:,0,:,:], targets[:,0,:,:])#)

            print("epoch:{:2d}  Lr: {:.5f} || Loss: {:.5f} || Cost Time: {} || mIOU: {:.5f} || Test_mIOU: {:.5f}".format(
                epoch+1, optimizer.param_groups[0]['lr'], epoch_loss, str(datetime.timedelta(seconds=int(time.time() - start_time))), miou/self.train_num, test_miou/self.test_num))
            # if (max_IOU < test_miou/self.test_num):
            #     max_IOU = test_miou/self.test_num
            torch.save(self.model, self.args.savedir)
        return max_IOU


def mIOU(outputs, targets):
    num_pic = outputs.shape[0]
    outputs = torch.abs(outputs).view(num_pic, -1).detach().cpu().numpy()
    targets = targets.view(num_pic, -1).detach().cpu().numpy()
    intersection = (outputs * targets).sum(1)
    union = outputs.sum(1) + targets.sum(1) + 1e-7
    iou = intersection / (union - intersection)
    return iou.sum()

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.loaddir).to(args.device)#Unet()#
    ##加载部分参数
    # save_model = torch.load('D:/Download/uunet3_0622.pth')
    # model_dict =  model.state_dict()
    # state_dict = {k:v for k,v in save_model.state_dict().items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    model.out.bias.data.fill_(-4.6)

    trainer = Trainer(args, model)
    iou = trainer.train()
    print("best perform IoU:", iou)
