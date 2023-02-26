# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:48:34 2023

@author: lab503
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore


class Generator(nn.Cell):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim

        self.deconv1 = nn.Conv2dTranspose(in_channels=int(self.rep_dim / (4 * 4)), out_channels=128, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        
        self.bn2d4 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.deconv2 = nn.Conv2dTranspose(in_channels=128, out_channels=64, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.deconv3 = nn.Conv2dTranspose(in_channels=64, out_channels=32, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
       
        self.bn2d6 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.deconv4 = nn.Conv2dTranspose(in_channels=32, out_channels=3, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.leaky_relu = nn.LeakyReLU(0.01)
    def construct(self, x):
     
        x = x.reshape(int(x.shape[0]), int(self.rep_dim / (4 * 4)), 4, 4)
        x = self.leaky_relu(x)
        x = self.deconv1(x)
        
        x = ops.interpolate(self.leaky_relu(self.bn2d4(x)), sizes=(8,8), mode='bilinear')
        
        x = self.deconv2(x)
        x = ops.interpolate(self.leaky_relu(self.bn2d5(x)), sizes=(16,16), mode='bilinear')
        x = self.deconv3(x)
        x = ops.interpolate(self.leaky_relu(self.bn2d6(x)), sizes=(32,32), mode='bilinear')
        x = self.deconv4(x)
        x = ops.sigmoid(x)
        return x


    
class Encoder(nn.Cell):
    def __init__(self, rep_dim=128):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2,2) 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size = 5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Dense(128 * 4 * 4, self.rep_dim, has_bias=False)
        self.leaky_relu = nn.LeakyReLU(0.01)
    def construct(self, x):
        
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(self.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(self.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(self.leaky_relu(self.bn2d3(x)))
        x = x.reshape(int(x.shape[0]), -1)
        x = self.fc1(x)
        return x

class Discriminatorxz(nn.Cell):
    def __init__(self, rep_dim=128):
        super(Discriminatorxz, self).__init__()
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2,2)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.rnd_layer1 = nn.Dense(self.rep_dim, 512)
        self.layer1 = nn.SequentialCell(
            nn.Dense(128*4*4 + 512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Dense(1024, 1)
        

    def construct(self, img, z):
        x = img.reshape(img.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(self.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(self.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(self.leaky_relu(self.bn2d3(x)))
        img_out = x.reshape(int(x.shape[0]), -1)
        
        
       
        z = z.view(z.shape[0], -1)
        z_out = self.rnd_layer1(z)
        
        
        
        out = ops.concat([img_out, z_out], axis=1)
        
        out = self.layer1(out)

        feature = out
        feature = feature.reshape(feature.shape[0], -1)

        d_out = self.layer2(out)
        
        return d_out,feature
    
class Discriminatorxx(nn.Cell):
    def __init__(self,rep_dim=128):
        super(Discriminatorxx, self).__init__()
        self.rep_dim = rep_dim
        
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)

        
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d4 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, has_bias=False, padding=2, pad_mode ='pad')
        self.bn2d6 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.layer1 = nn.SequentialCell(
            nn.Dense(128 * 4 * 4 * 2, 1024),
            nn.LeakyReLU(0.1)
        )
        
        self.layer2 = nn.Dense(1024, 1)

        
    def construct(self, img1, img2):
        x = img1.view(-1, 3, 32, 32)
        x = self.conv1(x)
        x = self.pool(self.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(self.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(self.leaky_relu(self.bn2d3(x)))
        img_out1 = x.view(int(x.size(0)), -1)
        
        x = img2.view(-1, 3, 32, 32)
        x = self.conv4(x)
        x = self.pool(self.leaky_relu(self.bn2d4(x)))
        x = self.conv5(x)
        x = self.pool(self.leaky_relu(self.bn2d5(x)))
        x = self.conv6(x)
        x = self.pool(self.leaky_relu(self.bn2d6(x)))
        img_out2 = x.view(int(x.size(0)), -1)
        
        
        
        out = ops.concat([img_out1, img_out2], axis=1)
        out = self.layer1(out)
        
        feature = out
        feature = feature.view(feature.size()[0], -1)

        d_out = self.layer2(out)
        
        return d_out,feature

