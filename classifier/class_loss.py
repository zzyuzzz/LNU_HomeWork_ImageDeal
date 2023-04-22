import torch
import torch.nn as nn

import numpy as np
import math
from torchvision import datasets


lambda_coord = 5
lambda_noobj = .5


def squareDiff(x:tuple, y:tuple):
    t = x - y
    return t[0] ** 2 + t[1] ** 2

def objInCell(box, truth_box):
    print(truth_box.shape)
    center_x = truth_box[0] + truth_box[2] / 2
    center_y = truth_box[1] + truth_box[3] / 2
    # center_x = center_x.numpy()
    # center_y = center_y.numpy()
    return box[0] <= center_x and box[0] + box[2] >= center_x \
            and box[1] <= center_y and box[1] + box[3] >= center_y

S = 7

def Loss(X, bbox):
    loss_coord = 0
    loss_noobj = 0
    loss_0 = 0
    for x in X:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                center_x = x[i, j][0] - x[i, j][2] / 2
                center_y = x[i, j][1] - x[i, j][3] / 2
                # print(x)
                for i in range(bbox.shape[0]):
                    y = bbox[i][0]
                    print("bbox",bbox)
                    print("bbox y ", y)
                    if objInCell((center_x, 
                                center_y,
                                x[i, j][2],x[i, j][3]
                                ), y):
                        loss_coord = loss_coord + squareDiff((center_x, center_y), (y[0], y[1]))
                        loss_coord = loss_coord + squareDiff((math.sqrt(x[i, j][2]),
                                                            math.sqrt(x[i, j][3])), 
                                                            (math.sqrt(y[2]),
                                                            math.sqrt(y[3])))
                        loss_0 = loss_0 + (1 - x[i, j, 4]) ** 2
                    else:
                        loss_noobj = loss_noobj + x[i, j, 4] ** 2

    return lambda_coord * loss_coord + lambda_noobj * loss_noobj + loss_0