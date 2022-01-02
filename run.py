# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 19:57:37 2021

@author: DELL
"""

import numpy as np
import time
import os
import sys
import segmentation_models_pytorch as smp
import torch
import cv2
from torchvision import transforms as T
import glob
import torch.utils.data as D

class OurDataset(D.Dataset):
    def __init__(self, TifArray):
        self.TifArray = [i for item in TifArray for i in item]
        self.len = len(TifArray)*len(TifArray[0])

        self.as_tensor = T.Compose([
            # 将numpy的ndarray转换成形状为(C,H,W)的Tensor格式,且/255归一化到[0,1.0]之间
            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        
        image = self.TifArray[index]
        image_A = image[:,:,0:3]
        image_B = image[:,:,3:6]
        image_B_fromA= style_transfer(image_B, image_A)
        image = np.concatenate((image_A,image_B_fromA),axis=2)
        
        return self.as_tensor(image)
    # 数据集数量
    def __len__(self):
        return self.len

def get_dataloader(TifArray, batch_size, shuffle=False, num_workers=8):
    dataset = OurDataset(TifArray)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader

def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:,:,i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:,:,i])
        target_image_fshift = np.fft.fftshift(target_image_f)
        
        change_length = 1
        source_image_fshift[int(h/2)-change_length:int(h/2)+change_length, 
                            int(h/2)-change_length:int(h/2)+change_length] = \
            target_image_fshift[int(h/2)-change_length:int(h/2)+change_length,
                                int(h/2)-change_length:int(h/2)+change_length]
            
        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)
        
        source_image_if[source_image_if>255] = np.max(source_image[:,:,i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1,0).swapaxes(1,2)
    
    out = out.astype(np.uint8)
    return out


#  tif裁剪（tif像素数据，裁剪边长）
def TifCroppingArray(img, SideLength):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int((img.shape[0] - SideLength * 2) / (512 - SideLength * 2))
    #  行上图像块数目
    RowNum = int((img.shape[1] - SideLength * 2) / (512 - SideLength * 2))
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * (512 - SideLength * 2) : i * (512 - SideLength * 2) + 512,
                          j * (512 - SideLength * 2) : j * (512 - SideLength * 2) + 512]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
    #  考虑到行列会有剩余的情况，向前裁剪一行和一列
    #  向前裁剪最后一列
    for i in range(ColumnNum):
        cropped = img[i * (512 - SideLength * 2) : i * (512 - SideLength * 2) + 512,
                      (img.shape[1] - 512) : img.shape[1]]
        TifArrayReturn[i].append(cropped)
    #  向前裁剪最后一行
    TifArray = []
    for j in range(RowNum):
        cropped = img[(img.shape[0] - 512) : img.shape[0],
                      j * (512-SideLength*2) : j * (512 - SideLength * 2) + 512]
        TifArray.append(cropped)
    #  向前裁剪右下角
    cropped = img[(img.shape[0] - 512) : img.shape[0],
                  (img.shape[1] - 512) : img.shape[1]]
    TifArray.append(cropped)
    TifArrayReturn.append(TifArray)
    #  列上的剩余数
    ColumnOver = (img.shape[0] - SideLength * 2) % (512 - SideLength * 2) + SideLength
    #  行上的剩余数
    RowOver = (img.shape[1] - SideLength * 2) % (512 - SideLength * 2) + SideLength
    return TifArrayReturn, RowOver, ColumnOver

#  获得结果矩阵
def Result(shape, TifArray, npyfile, RepetitiveLength, RowOver, ColumnOver):
    result = np.zeros(shape, np.uint8)
    #  j来标记行数
    j = 0  
    for i,img in enumerate(npyfile):
        #  最左侧一列特殊考虑，左边的边缘要拼接进去
        if(i % len(TifArray[0]) == 0):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 512 - RepetitiveLength, 0 : 512-RepetitiveLength] = img[0 : 512 - RepetitiveLength, 0 : 512 - RepetitiveLength]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                #  原来错误的
                #result[shape[0] - ColumnOver : shape[0], 0 : 512 - RepetitiveLength] = img[0 : ColumnOver, 0 : 512 - RepetitiveLength]
                #  后来修改的
                result[shape[0] - ColumnOver - RepetitiveLength: shape[0], 0 : 512 - RepetitiveLength] = img[512 - ColumnOver - RepetitiveLength : 512, 0 : 512 - RepetitiveLength]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength,
                       0:512-RepetitiveLength] = img[RepetitiveLength : 512 - RepetitiveLength, 0 : 512 - RepetitiveLength]   
        #  最右侧一列特殊考虑，右边的边缘要拼接进去
        elif(i % len(TifArray[0]) == len(TifArray[0]) - 1):
            #  第一行的要再特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 512 - RepetitiveLength, shape[1] - RowOver: shape[1]] = img[0 : 512 - RepetitiveLength, 512 -  RowOver: 512]
            #  最后一行的要再特殊考虑，下边的边缘要考虑进去
            elif(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0], shape[1] - RowOver : shape[1]] = img[512 - ColumnOver : 512, 512 - RowOver : 512]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength,
                       shape[1] - RowOver : shape[1]] = img[RepetitiveLength : 512 - RepetitiveLength, 512 - RowOver : 512]   
            #  走完每一行的最右侧，行数+1
            j = j + 1
        #  不是最左侧也不是最右侧的情况
        else:
            #  第一行的要特殊考虑，上边的边缘要考虑进去
            if(j == 0):
                result[0 : 512 - RepetitiveLength,
                       (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[0 : 512 - RepetitiveLength, RepetitiveLength : 512 - RepetitiveLength]         
            #  最后一行的要特殊考虑，下边的边缘要考虑进去
            if(j == len(TifArray) - 1):
                result[shape[0] - ColumnOver : shape[0],
                       (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength
                       ] = img[512 - ColumnOver : 512, RepetitiveLength : 512 - RepetitiveLength]
            else:
                result[j * (512 - 2 * RepetitiveLength) + RepetitiveLength : (j + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength,
                       (i - j * len(TifArray[0])) * (512 - 2 * RepetitiveLength) + RepetitiveLength : (i - j * len(TifArray[0]) + 1) * (512 - 2 * RepetitiveLength) + RepetitiveLength,
                       ] = img[RepetitiveLength : 512 - RepetitiveLength, RepetitiveLength : 512 - RepetitiveLength]
    return result


def test_bigImage(TifPath_A, TifPath_B, model_paths, ResultPath,RepetitiveLength):


    
    # big_image = readTif(TifPath)
    # big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)
    
    big_image_A = cv2.imread(TifPath_A, cv2.IMREAD_UNCHANGED)
    big_image_A = cv2.cvtColor(big_image_A, cv2.COLOR_BGR2RGB)
    big_image_B = cv2.imread(TifPath_B, cv2.IMREAD_UNCHANGED)
    big_image_B = cv2.cvtColor(big_image_B, cv2.COLOR_BGR2RGB)
    
    
#     big_image_B_fromA= style_transfer(big_image_B, big_image_A)
#     big_image = np.concatenate((big_image_A,big_image_B_fromA),axis=2)
    
    
    big_image = np.concatenate((big_image_A,big_image_B),axis=2)
    
    TifArray, RowOver, ColumnOver = TifCroppingArray(big_image, RepetitiveLength)
    
    trfm = T.Compose([
        T.ToTensor(),
        ])
    
    model = smp.UnetPlusPlus(
            # encoder_name="timm-resnest101e", 
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=6,
            decoder_attention_type="scse",
            classes=1,
            activation="sigmoid",
    )
    # 将模型加载到指定设备DEVICE上
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    model.to(DEVICE)
    
    predicts = []
    
    test_loader = get_dataloader(TifArray, batch_size=64)
    
    # weights = [1.5,1]
    for image in test_loader:
        
        model.to(DEVICE)
        output = np.zeros((image.shape[0],1,512,512))
        for model_path_ind, model_path in enumerate(model_paths):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            with torch.no_grad():
                
                image = image.cuda()
                
#                 output1 = model(image).cpu().numpy()
            
#                 output2 = model(torch.flip(image, [0, 3]))
#                 output2 = torch.flip(output2, [3, 0]).cpu().numpy()
            
#                 output3 = model(torch.flip(image, [0, 2]))
#                 output3 = torch.flip(output3, [2, 0]).cpu().numpy()
                    
#                 output4 = model(torch.flip(torch.flip(image, [0, 3]), [0, 2]))
#                 output4 = torch.flip(torch.flip(output4, [2, 0]), [3, 0]).cpu().numpy()
                
                output1 = model(image).cpu().data.numpy()
                
            # output += output1 + output2 + output3 + output4
            output += output1
            # output += output1*weights[model_path_ind]
        
        # output = output / (len(model_paths)*4)
        output = output / (len(model_paths)*1)
        # output = output / 2.5
        
        # output.shape: batch_size,classes,512,512
        for i in range(output.shape[0]):
            pred = output[i]
            threshold = 0.21
            pred[pred>=threshold] = 1
            pred[pred<threshold] = 0
            pred = np.uint8(pred)
            pred = pred.reshape((512,512))
            predicts.append((pred))
    
    
#     for i in range(len(TifArray)):
#         for j in range(len(TifArray[0])):
#             image = TifArray[i][j]
            
#             image_A = image[:,:,0:3]
#             image_B = image[:,:,3:6]
#             image_B_fromA= style_transfer(image_B, image_A)
#             image = np.concatenate((image_A,image_B_fromA),axis=2)
            
#             image = trfm(image)
#             image = image.cuda()[None]
#             pred = np.zeros((1,1,512,512))
#             for model_path in model_paths:
#                 model.load_state_dict(torch.load(model_path))
#                 model.eval()
                
#                 with torch.no_grad():
                    
#                     pred1 = model(image).cpu().numpy()
            
# #                     pred2 = model(torch.flip(image, [0, 3]))
# #                     pred2 = torch.flip(pred2, [3, 0]).cpu().numpy()
            
# #                     pred3 = model(torch.flip(image, [0, 2]))
# #                     pred3 = torch.flip(pred3, [2, 0]).cpu().numpy()
                    
# #                     pred4 = model(torch.flip(torch.flip(image, [0, 3]), [0, 2]))
# #                     pred4 = torch.flip(torch.flip(pred4, [2, 0]), [3, 0]).cpu().numpy()
                    
#                     # pred += pred1 + pred2 + pred3 + pred4
#                     pred += pred1
                    
            # pred = pred / (len(model_paths) * 1)
            # threshold = 0.21
            # pred[pred>=threshold] = 1
            # pred[pred<threshold] = 0
            # pred = pred.astype(np.uint8)
            # pred = pred.reshape((512,512))
            # predicts.append((pred))
    
    #保存结果predictspredicts
    result_shape = (big_image.shape[0], big_image.shape[1])
    result_data = Result(result_shape, TifArray, predicts, RepetitiveLength, RowOver, ColumnOver)
    cv2.imwrite(ResultPath, result_data)


if __name__ == "__main__":
    start_time = time.time()
    model_paths = [
        "/save_dir/UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_120_fold_2.pth",
        "/save_dir/UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_120_fold_4.pth",
        # "UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_105_fold_1.pth",
        # "UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_105_fold_2.pth",
        # "UnetPlusPlus_efficientnetb0_FFT_copy_Agu_edge_105_fold_4.pth",
        ]
    RepetitiveLength = 100
    # output_dir = r"E:\WangZhenQing\2021ShengTeng\data\test_AB_big"
    # input_dir = r"E:\WangZhenQing\2021ShengTeng\data\test_AB_big"
    output_dir = sys.argv[2]
    input_dir = sys.argv[1]
    image_A_paths = glob.glob(input_dir+'/A/*.tif')
    image_B_paths = glob.glob(input_dir+'/B/*.tif')
    for image_A_path in image_A_paths:
        image_B_path = image_A_path.replace("/A","/B")
        
        image_name = image_A_path.split("/")[-1]
        ResultPath = os.path.join(output_dir, image_name.replace("tif","png"))
        
        test_bigImage(image_A_path, image_B_path, model_paths, ResultPath,RepetitiveLength)
    print((time.time()-start_time))
    
