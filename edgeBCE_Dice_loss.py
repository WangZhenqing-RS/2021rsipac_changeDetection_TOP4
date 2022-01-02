# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:02:34 2021

@author: DELL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss


def edgeBCE_Dice_loss(pred, target, edge):
    
    # diceloss在一定程度上可以缓解类别不平衡,但是训练容易不稳定
    DiceLoss_fn = DiceLoss(mode='binary',
                           from_logits=False)
    # 交叉熵
    BinaryCrossEntropy_fn = nn.BCELoss(reduction='none')
    
    # 
    edge_weight = 4.
    loss_bce = BinaryCrossEntropy_fn(pred, target)
    loss_dice = DiceLoss_fn(pred, target)
    edge[edge == 0] = 1.
    edge[edge == 255] = edge_weight
    loss_bce *= edge
    # OHEM
    loss_bce_,ind = loss_bce.contiguous().view(-1).sort()
    min_value = loss_bce_[int(0.5*loss_bce.numel())]
    loss_bce = loss_bce[loss_bce>=min_value]
    loss_bce = loss_bce.mean()
    loss = loss_bce + loss_dice
    return loss


# if __name__ == '__main__':
#     target=torch.ones((2,1,256,256),dtype=torch.float32)
#     input=(torch.ones((2,1,256,256))*0.9)
#     input[0,0,0,0] = 0.99
#     loss=edgeBCE_Dice_loss(input,target,target*255)
#     print(loss)

