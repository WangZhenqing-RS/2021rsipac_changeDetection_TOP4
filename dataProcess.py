# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""

import torch.utils.data as D
from torchvision import transforms as T
import random
import numpy as np
import torch
import cv2
import albumentations as A
from style_transfer import style_transfer
from data_agu import data_agu
from edge_utils import mask_to_onehot, onehot_to_binary_edges
from copy_paste import copy_paste_self, copy_paste

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集Iou
def cal_val_iou(model, loader):
    val_iou = []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        iou = cal_iou(output, target)
        val_iou.append(iou)
    return val_iou

#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集f1
def cal_val_f1(model, loader):
    TP_sum, FN_sum, FP_sum = [], [], []
    # 需要加上model.eval()
    # 否则的话，有输入数据，即使不训练，它也会改变权值
    # 这是model中含有BN和Dropout所带来的的性质
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        output[output>=0.5] = 1
        output[output<0.5] = 0
        TP, FN, FP = cal_f1(output, target)
        TP_sum.append(TP)
        FN_sum.append(FN)
        FP_sum.append(FP)
    
    p = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FP_sum) + 0.000001)
    r = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FN_sum) + 0.000001)
    val_f1 = 2 * r * p / (r + p + 0.000001)
    return val_f1

# 计算IoU
def cal_iou(pred, mask, c=1):
    iou_result = []
    # for idx in range(c):
    idx = c
    p = (mask == idx).int().reshape(-1)
    t = (pred == idx).int().reshape(-1)
    uion = p.sum() + t.sum()
    overlap = (p*t).sum()
    #  0.0001防止除零
    iou = 2*overlap/(uion + 0.000001)
    iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

# 计算IoU
def cal_f1(pred, mask, c=1):
    # f1_result = []
    # TP    predict 和 label 同时为1
    TP = ((pred == 1) & (mask == 1)).cpu().sum()
    # TN    predict 和 label 同时为0
    # TN = ((pred == 0) & (mask == 0)).cpu().sum()
    # FN    predict 0 label 1
    FN = ((pred == 0) & (mask == 1)).cpu().sum()
    # FP    predict 1 label 0
    FP = ((pred == 1) & (mask == 0)).cpu().sum()
    
    # p = TP / (TP + FP + 0.000001)
    # r = TP / (TP + FN + 0.000001)
    # f1 = 2 * r * p / (r + p + 0.000001)

    # f1_result.append(f1)
    return TP, FN, FP

class OurDataset(D.Dataset):
    def __init__(self, image_A_paths, image_B_paths, label_paths, mode):
        self.image_A_paths = image_A_paths
        self.image_B_paths = image_B_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_A_paths)

        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        image_A = cv2.imread(self.image_A_paths[index],cv2.IMREAD_UNCHANGED)
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.imread(self.image_B_paths[index],cv2.IMREAD_UNCHANGED)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2RGB)

        
        if self.mode == "train":
            
            # FFT风格统一变换
            if np.random.random() < 0.5:
                image_B = style_transfer(image_B, image_A)
            else:
                image_A = style_transfer(image_A, image_B)
            
            label = cv2.imread(self.label_paths[index],0)
            
            if(np.random.random()<0.5):
                # 复制粘贴增强
                label[label==1]=255
                random_value = np.random.random()
                copy_index = int(np.random.random()*len(self.image_A_paths))
                image_copy_A = cv2.imread(self.image_A_paths[copy_index],cv2.IMREAD_UNCHANGED)
                label_copy = cv2.imread(self.label_paths[copy_index],0)
                _, image_A = copy_paste(image_copy_A, label_copy, image_A, label, random_value)
                image_copy_B = cv2.imread(self.image_B_paths[copy_index],cv2.IMREAD_UNCHANGED)
                label, image_B = copy_paste(image_copy_B, label_copy, image_B, label, random_value)
                label[label==255]=1
            
            # 普通数据增强
            image_A, image_B, label = data_agu(image_A, image_B, label)
            
            
            
            # 两时相图像叠加
            image = np.concatenate((image_A,image_B),axis=2)
            
            # 获取正样本边缘->服务于边缘加权bce
            oneHot_label = mask_to_onehot(label,2) #edge=255,background=0
            edge = onehot_to_binary_edges(oneHot_label,2,2)
            # 消去图像边缘
            edge[:2, :] = 0
            edge[-2:, :] = 0
            edge[:, :2] = 0
            edge[:, -2:] = 0
            
            label = label.reshape((1,) + label.shape)
            edge = edge.reshape((1,) + edge.shape)
            return self.as_tensor(image), label.astype(np.int64), edge
            # label[label==1] =255
            # return image_A, image_B, label, edge
        elif self.mode == "val":
            
            # FFT风格统一变换
            if np.random.random() < 0.5:
                image_B = style_transfer(image_B, image_A)
            else:
                image_A = style_transfer(image_A, image_B)
            
            image = np.concatenate((image_A,image_B),axis=2)
            
            label = cv2.imread(self.label_paths[index],0)
            
            label = label.reshape((1,) + label.shape)
            return self.as_tensor(image), label.astype(np.int64)
        elif self.mode == "test":
            # FFT风格统一变换
            image_B_fromA = style_transfer(image_B, image_A)
            image_A_BfromA = np.concatenate((image_A,image_B_fromA),axis=2)
            image_A_fromB= style_transfer(image_A, image_B)
            image_AfromB_B = np.concatenate((image_A_fromB,image_B),axis=2)
            return self.as_tensor(image_A_BfromA), self.as_tensor(image_AfromB_B), self.image_A_paths[index]
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(image_A_paths, image_B_paths, label_paths, mode, batch_size, 
                   shuffle, num_workers):
    dataset = OurDataset(image_A_paths, image_B_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def split_train_val(image_A_paths, image_B_paths, label_paths, val_index=0):
    # 分隔训练集和验证集
    train_image_A_paths, train_image_B_paths, train_label_paths, val_image_A_paths, val_image_B_paths, val_label_paths = [], [], [], [], [], []
    for i in range(len(image_A_paths)):
        # 训练验证4:1,即每5个数据的第val_index个数据为验证集
        if i % 5 == val_index:
            val_image_A_paths.append(image_A_paths[i])
            val_image_B_paths.append(image_B_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_A_paths.append(image_A_paths[i])
            train_image_B_paths.append(image_B_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of train images: ", len(train_image_A_paths))
    print("Number of val images: ", len(val_image_A_paths))
    return train_image_A_paths, train_image_B_paths, train_label_paths, val_image_A_paths, val_image_B_paths, val_label_paths


# import glob
# import matplotlib.pyplot as plt

# image_A_paths = glob.glob(r"../data/train/A/*.tif")
# image_B_paths = glob.glob(r"../data/train/B/*.tif")
# label_paths = glob.glob(r"../data/train/label/*.png")

# dataset = OurDataset(image_A_paths, image_B_paths, label_paths, "train")
# a = dataset[40]
# plt.subplot(141)
# plt.imshow(a[0])
# plt.subplot(142)
# plt.imshow(a[1])
# plt.subplot(143)
# plt.imshow(a[2])
# plt.subplot(144)
# plt.imshow(a[3])