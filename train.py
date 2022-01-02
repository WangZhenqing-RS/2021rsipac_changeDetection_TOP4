# -*- coding: utf-8 -*-
"""
@author: wangzhenqing
ref:
1. 
2.https://github.com/DLLXW/data-science-competition/tree/main/%E5%A4%A9%E6%B1%A0
3.https://github.com/JasmineRain/NAIC_AI-RS/tree/ec70861e2a7f3ba18b3cc8bad592e746145088c9
"""
import numpy as np
import torch
import warnings
import time
from dataProcess import get_dataloader, cal_val_f1, split_train_val
import segmentation_models_pytorch as smp
import glob
# from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
# from pytorch_toolbelt import losses as L
# import torch.nn as nn
from edgeBCE_Dice_loss import edgeBCE_Dice_loss
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# 忽略警告信息
warnings.filterwarnings('ignore')
# cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
torch.backends.cudnn.enabled = True

# 将模型加载到指定设备DEVICE上
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def train(EPOCHES, BATCH_SIZE, train_image_A_paths, train_image_B_paths,
          train_label_paths, val_image_A_paths, val_image_B_paths,
          val_label_paths, channels, optimizer_name, model_path, early_stop):
    
    train_loader = get_dataloader(train_image_A_paths, train_image_B_paths,
                                  train_label_paths, "train", BATCH_SIZE,
                                  shuffle=True, num_workers=8)
    valid_loader = get_dataloader(val_image_A_paths, val_image_B_paths, 
                                  val_label_paths, "val", BATCH_SIZE,
                                  shuffle=False, num_workers=8)
    
    # 定义模型,优化器,损失函数
    model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b0", 
            # encoder_weights="imagenet",
            encoder_weights=None,
            in_channels=3,
            decoder_attention_type="scse",
            classes=1,
            activation="sigmoid",
    )
    model.encoder.load_state_dict(torch.load("efficientnet-b0-355c32eb.pth"))
    model.encoder.set_in_channels(channels)
    model.to(DEVICE);
    # model.load_state_dict(torch.load(r"E:\WangZhenQing\2021ShengTeng\model\UnetPlusPlus_efficientnetb3_FFT_Agu_edge_SGD_300_fold_0.pth"))
    # # 采用AdamM优化器
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0.1)
    
    
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
            )
    
    # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True,min_lr=1e-6)


    # # 损失函数采用SoftCrossEntropyLoss+DiceLoss
    # DiceLoss_fn=DiceLoss(mode='binary',
    #                      from_logits=False)
    # BinaryCrossEntropy_fn=nn.BCELoss()
    # loss_fn = L.JointLoss(first=DiceLoss_fn, second=BinaryCrossEntropy_fn,
    #                       first_weight=0.5, second_weight=0.5).cuda()
    
    loss_fn = edgeBCE_Dice_loss
    header = r'Epoch/EpochNum | TrainLoss | ValidF1 | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.5f} | {:9.5f} | {:9.2f}'
    print(header)
 
    # 记录当前验证集最优f1,以判定是否保存当前模型
    best_f1 = 0
    best_f1_epoch = 0
    train_loss_epochs, val_f1_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, EPOCHES+1):
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE);
        for batch_index, (image, target, edge) in enumerate(train_loader):
            
            image, target, edge= image.to(DEVICE), target.to(DEVICE), edge.to(DEVICE)
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            output = model(image)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            # 求解该batch的loss
            loss = loss_fn(output, target, edge)
            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            losses.append(loss.item())
        
        # 计算验证集f1
        val_f1 = cal_val_f1(model, valid_loader)
        scheduler.step()
        # scheduler.step(val_f1)
        # 输出验证集每类f1
        # print('\t'.join(np.stack(val_f1).mean(0).round(3).astype(str))) 
        # 保存当前epoch的train_loss.val_f1.lr_epochs
        train_loss_epochs.append(np.array(losses).mean())
        val_f1_epochs.append(val_f1)
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 输出进程
        print(raw_line.format(epoch, EPOCHES, np.array(losses).mean(), 
                              np.mean(val_f1), 
                              (time.time()-start_time)/60**1), end="")    
        if best_f1 < val_f1:
            best_f1 = val_f1
            best_f1_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("  valid f1 is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_f1_epoch) >= early_stop:
                break
    return train_loss_epochs, val_f1_epochs, lr_epochs
    
# 不加主函数这句话的话,Dataloader多线程加载数据会报错
if __name__ == '__main__':
    start_time = time.time()
    for fold_k in [0,1,2,4]:
        EPOCHES = 105
        # BATCH_SIZE = 8
        BATCH_SIZE = 6
        # fold_k = 0
        print("fold = ", fold_k)
        # 不同机器可能会导致glob得到的路径顺序不一致
        # image_A_paths = glob.glob("/train/A/*.tif")
        image_A_paths = []
        with open("image_A_paths.txt", "r") as f:
            for line in f.readlines():
                if "\r\n" in line:
                    line = line.strip('\r\n')
                elif "\n" in line:
                    line = line.strip('\n')
                image_A_paths.append(line)

        # image_B_paths = glob.glob(r"../data/train/B/*.tif")
        # label_paths = glob.glob(r"../data/train/label/*.png")
        image_B_paths, label_paths = [],[]
        for image_A_path in image_A_paths:
            image_B_path = image_A_path.replace("A","B")
            image_B_paths.append(image_B_path)
            label_path = image_A_path.replace("A","label").replace("tif","png")
            label_paths.append(label_path)
        
        train_image_A_paths, train_image_B_paths, train_label_paths, val_image_A_paths, val_image_B_paths, val_label_paths =  split_train_val(image_A_paths,
                                                                                                                                              image_B_paths,
                                                                                                                                              label_paths,
                                                                                                                                              val_index=fold_k)
        
        channels = 6
        
        optimizer_name = "adamw"
        model_path = "UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_105_fold_{0}.pth".format(fold_k)
        early_stop = 100
        train_loss_epochs, val_f1_epochs, lr_epochs = train(EPOCHES, 
                                                              BATCH_SIZE, 
                                                              train_image_A_paths, 
                                                              train_image_B_paths,
                                                              train_label_paths,
                                                              val_image_A_paths,
                                                              val_image_B_paths,
                                                              val_label_paths,
                                                              channels, 
                                                              optimizer_name,
                                                              model_path,
                                                              early_stop)
        
    print((time.time()-start_time)/60/60)
    # if(True):    
    #     import matplotlib.pyplot as plt
    #     epochs = range(1, len(train_loss_epochs) + 1)
    #     plt.plot(epochs, train_loss_epochs, 'r', label = 'train loss')
    #     plt.plot(epochs, val_f1_epochs, 'b', label = 'val f1')
    #     plt.title('train loss and val f1')
    #     plt.legend()
    #     plt.savefig(r"../plt/train loss and val f1.png",dpi = 300)
    #     plt.figure()
    #     plt.plot(epochs, lr_epochs, 'r', label = 'learning rate')
    #     plt.title('learning rate') 
    #     plt.legend()
    #     plt.savefig(r"../plt/learning rate.png", dpi = 300)
    #     plt.show() 
