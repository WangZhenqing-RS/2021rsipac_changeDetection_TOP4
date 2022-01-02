# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:11:24 2021

@author: wangzhenqing

ref：
https://github.com/liaochengcsu/road_segmentation_pytorch/blob/main/data_agu.py
"""
import cv2
import numpy as np

# 随机调节色调、饱和度值
def randomHueSaturationValue(image_A, image_B, hue_shift_limit=(-30, 30),
                             sat_shift_limit=(-5, 5),
                             val_shift_limit=(-15, 15), ratio=1):
    if np.random.random() < ratio:
        image_A = cv2.cvtColor(image_A, cv2.COLOR_BGR2HSV)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_BGR2HSV)
        h_A, s_A, v_A = cv2.split(image_A)
        h_B, s_B, v_B = cv2.split(image_B)
        # hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        # hue_shift = np.uint8(hue_shift)
        # h_A += hue_shift
        # h_B += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s_A = cv2.add(s_A, sat_shift)
        s_B = cv2.add(s_B, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v_A = cv2.add(v_A, val_shift)
        v_B = cv2.add(v_B, val_shift)
        image_A = cv2.merge((h_A, s_A, v_A))
        image_B = cv2.merge((h_B, s_B, v_B))
        image_A = cv2.cvtColor(image_A, cv2.COLOR_HSV2BGR)
        image_B = cv2.cvtColor(image_B, cv2.COLOR_HSV2BGR)
    return image_A, image_B

# 随机移位旋转
def randomShiftScaleRotate(image_A, image_B, mask,
                           shift_limit=(-0.1, 0.1),
                           scale_limit=(-0.1, 0.1),
                           aspect_limit=(-0.1, 0.1),
                           rotate_limit=(-0, 0),
                           borderMode=cv2.BORDER_CONSTANT, ratio=0.5):
    if np.random.random() < ratio:
        height, width, channel = image_A.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image_A = cv2.warpPerspective(image_A, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        image_B = cv2.warpPerspective(image_B, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image_A, image_B, mask


def randomHorizontalFlip(image_A, image_B, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = cv2.flip(image_A, 1)
        image_B = cv2.flip(image_B, 1)
        mask = cv2.flip(mask, 1)

    return image_A, image_B, mask


def randomVerticleFlip(image_A, image_B, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = cv2.flip(image_A, 0)
        image_B = cv2.flip(image_B, 0)
        mask = cv2.flip(mask, 0)

    return image_A, image_B, mask


def randomRotate90(image_A, image_B, mask, ratio=0.5):
    if np.random.random() < ratio:
        image_A = np.rot90(image_A).copy()
        image_B = np.rot90(image_B).copy()
        mask = np.rot90(mask).copy()

    return image_A, image_B, mask


def data_agu(image_A, image_B, label):
    
    # image_A = cv2.imread("15_A.tif")
    # image_B = cv2.imread("15_B.tif")
    # label = cv2.imread("15.png")
    
    image_A, image_B = randomHueSaturationValue(image_A, image_B)
    
    image_A, image_B, label = randomShiftScaleRotate(image_A, image_B, label)
    
    image_A, image_B, label = randomHorizontalFlip(image_A, image_B, label)
    image_A, image_B, label = randomVerticleFlip(image_A, image_B, label)
    image_A, image_B, label = randomRotate90(image_A, image_B, label)
    
    # label[label==1] = 255
    # image_a_b_label = np.hstack((image_A, image_B, label))
    # cv2.imwrite("15_a_b_1.png", image_a_b_label)
    
    return image_A, image_B, label