import torch
import torch.nn as nn

import numpy as np
import math


class ConvWithActiv(nn.Module):
    def __init__(self, conv_in, conv_out, conv_kernelsz,
                  conv_stride, conv_padding) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, conv_kernelsz, conv_stride, conv_padding),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.layers(x)




class ConvWithPooling(nn.Module):
    def __init__(self, conv_num, conv_in, conv_out, conv_kernelsz,
                  conv_stride, conv_padding, pooling_kernelsz, 
                  pooling_stride) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[ConvWithActiv(conv_in[i], conv_out[i], conv_kernelsz[i], conv_stride[i], 
                            conv_padding[i]) for i in range(conv_num)],
            nn.MaxPool2d(pooling_kernelsz, pooling_stride)
        )

    def forward(self, x):
        return self.layers(x)





class Darknet19(nn.Module):
    def __init__(self, size=224) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvWithPooling(1, [3], [32], [3], [1], ['same'], 2, 2),
            ConvWithPooling(1, [32], [64], [3], [1], ['same'], 2, 2),
            ConvWithPooling(1, [64, 128, 64], [128, 64, 128], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),
            ConvWithPooling(1, [128, 256, 128], [256, 128, 256], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),
            ConvWithPooling(1, [256, 512, 256, 512, 256], [512, 256, 512, 256, 512],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 2, 2),
            ConvWithPooling(1, [512, 1024, 512, 1024, 512, 1024], 
                               [1024, 512, 1024, 512, 1024],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 1, 1),
        )

        self.layers1 = nn.Sequential(
            nn.Conv2d(1024, 257, 1, padding='same'),
            nn.AvgPool2d((7, 7) if size==224 else (14, 14)),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = self.layers1(x)
        x = nn.LogSoftmax(1)(torch.squeeze(x))
        return x 
    

class ResDarknet19(nn.Module):
    def __init__(self, size=224) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvWithPooling(1, [3], [32], [3], [1], ['same'], 2, 2),
            ConvWithPooling(1, [32], [64], [3], [1], ['same'], 2, 2),)
        self.layers1 = nn.Sequential(
            ConvWithPooling(1, [64, 128, 64], [128, 64, 128], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),
            ConvWithPooling(1, [128, 256, 128], [256, 128, 256], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),)
        self.layers2 = nn.Sequential(
            ConvWithPooling(1, [256, 512, 256, 512, 256], [512, 256, 512, 256, 512],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 2, 2),)
        self.layers3 = nn.Sequential(
            ConvWithPooling(1, [512, 1024, 512, 1024, 512, 1024], 
                               [1024, 512, 1024, 512, 1024],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 1, 1),
        )

        self.layers4 = nn.Sequential(
            nn.Conv2d(1024, 257, 1, padding='same'),
            nn.AvgPool2d((7, 7) if size==224 else (14, 14)),
        )
        
    def forward(self, x):
        y = self.layers(x)
        # a = [nn.functional.interpolate(x, tuple(y.shape[2:]))]
        x = self.layers1(torch.cat([y,  *[nn.functional.interpolate(x, tuple(y.shape[2:]))]], 1))
        y = self.layers2(torch.cat([x , *[nn.functional.interpolate(y, tuple(x.shape[2:]))]], 1))
        x = self.layers3(torch.cat([y , *[nn.functional.interpolate(x, tuple(y.shape[2:]))]], 1))
        x = self.layers4(x)
        x = nn.LogSoftmax(1)(torch.squeeze(x))
        return x 
    



##############

class ConvWithActivB(nn.Module):
    def __init__(self, conv_in, conv_out, conv_kernelsz,
                  conv_stride, conv_padding) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(conv_in, conv_out, conv_kernelsz, conv_stride, conv_padding),
            # nn.BatchNorm2d(conv_out),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        return self.layers(x)




class ConvWithPoolingB(nn.Module):
    def __init__(self, conv_num, conv_in, conv_out, conv_kernelsz,
                  conv_stride, conv_padding, pooling_kernelsz, 
                  pooling_stride, hasBN=True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[ConvWithActivB(conv_in[i], conv_out[i], conv_kernelsz[i], conv_stride[i], 
                            conv_padding[i]) for i in range(conv_num)],
            nn.BatchNorm2d(conv_out[-1]),
            nn.MaxPool2d(pooling_kernelsz, pooling_stride)
        )

    def forward(self, x):
        return self.layers(x)





class BNDarknet19(nn.Module):
    def __init__(self, size=224) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvWithPoolingB(1, [3], [32], [3], [1], ['same'], 2, 2),
            ConvWithPoolingB(1, [32], [64], [3], [1], ['same'], 2, 2),
            ConvWithPoolingB(1, [64, 128, 64], [128, 64, 128], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),
            ConvWithPoolingB(1, [128, 256, 128], [256, 128, 256], [3, 1, 3],
                             [1, 1, 1], ['same', 'same', 'same'], 2, 2),
            ConvWithPoolingB(1, [256, 512, 256, 512, 256], [512, 256, 512, 256, 512],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 2, 2),
            ConvWithPoolingB(1, [512, 1024, 512, 1024, 512, 1024], 
                               [1024, 512, 1024, 512, 1024],
                             [3, 1, 3, 1, 3], [1, 1, 1, 1, 1],
                            ['same', 'same', 'same', 'same', 'same'], 1, 1),
        )

        self.layers1 = nn.Sequential(
            nn.Conv2d(1024, 257, 1, padding='same'),
            nn.AvgPool2d((7, 7) if size==224 else (14, 14)),
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = self.layers1(x)
        x = torch.squeeze(x)
        return x 