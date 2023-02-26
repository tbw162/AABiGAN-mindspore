# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:35:26 2023

@author: lab503
"""

import argparse
import os
import numpy as np
import math
import tqdm
import copy
import pandas as pd
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings
import mindspore.nn as nn
import mindspore
import mindspore.ops as ops
a = 1
b = 0 
c = 0.75

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=200, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--normal_digit", type=int, default=0, help="noraml class")
parser.add_argument("--auxiliary_digit", type=int, default=1, help="abnormal aviliable during training process")
parser.add_argument("--gpu", type=str, default='3', help="gpu_num")
parser.add_argument("--dataset", type=str, default='MNIST', help="choice of dataset(CIFAR,F-MNIST,MNIST)")
parser.add_argument("--dir", type=str, default='/summary//', help="save dir")
parser.add_argument("--name", type=str, default='result', help="file name")
parser.add_argument("--gamma_l", type=float, default=0.2, help="ratio of auxiliary data")
parser.add_argument("--gamma_p", type=float, default=0, help="ratio of pollution data")
parser.add_argument("--k", type=float, default=1, help="the number of categories of the anomalous data")
parser.add_argument("--test_threshold", type=int,default=0,help="")
parser.add_argument("--error", type = int, default =0,help="")
opt = parser.parse_args()
print(opt.gamma_l)
pi = (1-opt.gamma_p-0.05*opt.error)
bn = (2*pi-1)/(1+pi)
import mindspore.context as context

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
if(opt.k<=1):
    seed = 12
else:
    seed = opt.auxiliary_digit
if(opt.dataset == 'CIFAR'):
    from dataset.cifarset_ms import create_loader 
    train_pos, train_neg, val_loader, test_loader = create_loader(opt)
if(opt.dataset=='CIFAR'):
    import model.arch_cifar_ms as arch
generator = arch.Generator()
discriminator = arch.Discriminatorxz()
encoder = arch.Encoder()

optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=0.0001, beta1=0.5, beta2=0.9)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=0.00025, beta1=0.5, beta2=0.9)
optimizer_E = nn.Adam(encoder.trainable_params(),learning_rate=0.0001,beta1 = 0.5,beta2=0.9)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')
optimizer_E.update_parameters_name('optim_e')

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
auc_re = pd.DataFrame()
best_val_recon = 0
best_test_recon = 0
best_val_zs = 0
best_test_zs = 0
import time 
z_score_test = []
test_features_all = []
adversarial_loss = nn.MSELoss()

def generator_forward(img_pos,img_neg,z_out_real,z_out_fake,valid):#生成器损失计算
    
    
    z = ops.concat([z_out_real,z_out_fake])
    gen = generator(z)
   
    gen_imgs_real = gen[:img_pos.shape[0]]
    gen_imgs_fake = gen[img_pos.shape[0]:]
    g_loss = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake)[0], c*valid)#+(1/3)*cycle_loss

    return g_loss,gen_imgs_real,gen_imgs_fake

def discriminator_forward(img_pos,gen_imgs_fake,img_neg,z_out_real,z_out_fake,z_out_neg,gen_imgs_real,valid):#判别器损失计算
    D_pos_xz = adversarial_loss(discriminator(img_pos,z_out_real)[0], a*valid)
    D_fake_xz = adversarial_loss(discriminator(gen_imgs_fake,z_out_fake)[0], b*valid)
    D_neg_xz = adversarial_loss(discriminator(img_neg,z_out_neg)[0],bn*(ops.ones((img_neg.shape[0], 1),mindspore.float32)))
    d_loss_xz = D_pos_xz+D_fake_xz+D_neg_xz
    
    
    d_loss = d_loss_xz
    
    return d_loss

def encoder_forward(img_pos,img_neg):#编码器损失计算
    img = ops.concat([img_pos,img_neg])
    #print(img.shape)
    z_out = encoder(img)
    #print(z_out.shape)
    z_out_real = z_out[:img_pos.shape[0]]
    z_out_neg = z_out[img_pos.shape[0]:]
    e_loss = adversarial_loss(discriminator(img_pos,z_out_real)[0],c*valid)+ adversarial_loss(discriminator(img_neg,z_out_neg)[0], c*(ops.ones((img_neg.shape[0], 1),mindspore.float32)))#+(1/3)*cycle_loss
    return e_loss, z_out_real, z_out_neg
grad_discriminator_fn = ops.value_and_grad(discriminator_forward, None,
                                           optimizer_D.parameters,has_aux=False)

grad_generator_fn = ops.value_and_grad(generator_forward,None,optimizer_G.parameters,has_aux=True)
grad_encoder_fn = ops.value_and_grad(encoder_forward,None,optimizer_E.parameters,has_aux=True)
train_pos = train_pos.create_tuple_iterator()
train_neg = train_neg.create_tuple_iterator()

from testing_ms import test_eva
#eva_dic= test_eva(generator,encoder,discriminator,0,val_loader,test_loader,opt)
for epoch in range(opt.n_epochs):
    start = time.time()
    i = 0
    dxx_list = []
    dxz_list = []
    for (batch_pos,batch_neg) in zip(train_pos,cycle(train_neg)):
    
        discriminator.set_train()
        generator.set_train()
        encoder.set_train()
        
    
        i+=1
        
        img_pos = batch_pos[0]
        #print(img_pos.shape)
        img_neg = batch_neg[0]
        #optimizer_D.zero_grad()
        target_pos = batch_pos[1]
        target_neg = batch_neg[1]        

        valid = ops.ones((img_pos.shape[0], 1), mindspore.float32)
        fake = ops.zeros((img_pos.shape[0], 1), mindspore.float32)


        z_out_fake = ops.StandardNormal()((img_pos.shape[0], opt.latent_dim))



        #print(img_pos.shape)
        img = ops.concat([img_pos,img_neg])
        #print(img.shape)
        z_out = encoder(img)
        #print(z_out.shape)
        z_out_real = z_out[:img_pos.shape[0]]
        z_out_neg = z_out[img_pos.shape[0]:]

        z = ops.concat([z_out_real,z_out_fake])
        gen = generator(z)

        gen_imgs_real = gen[:img_pos.shape[0]]
        gen_imgs_fake = gen[img_pos.shape[0]:]
      
        
        d_loss, d_grads = grad_discriminator_fn(img_pos,gen_imgs_fake,img_neg,z_out_real,z_out_fake,z_out_neg,gen_imgs_real,valid)
        optimizer_D(d_grads)
    
        #print(img_pos.shape)
        
        
        #print(discriminator(img_pos,z_out_real))
        (g_loss,gen_imgs_real,gen_imgs_fake), g_grads = grad_generator_fn(img_pos,img_neg,z_out_real,z_out_fake,valid)
        
        optimizer_G(g_grads)
        
        
        (e_loss,z_out_real, z_out_neg), e_grads = grad_encoder_fn(img_pos,img_neg)
       
        optimizer_E(e_grads)
        
        
        
        

        
        #cycle_loss = adversarial_loss(discriminator(img_pos,img_pos,'xx')[0],c*valid)+adversarial_loss(discriminator(img_pos,gen_imgs_real,'xx')[0],c*valid)+adversarial_loss(discriminator(img_neg,img_neg,'xx')[0],c*(ops.ones([img_neg.size(0), 1]),mindspore.float32))
       
        
  
            
        
       
        
    
        
    
        discriminator.set_train(False)
        generator.set_train(False)
        encoder.set_train(False)
        recon_pos = ((generator(encoder(img_pos))-img_pos)**2).sum(axis=(1,2,3))
        recon_pos = recon_pos.mean()
        
        recon_neg = ((generator(encoder(img_neg))-img_neg)**2).sum(axis=(1,2,3))
        recon_neg = recon_neg.mean()
        
        print(
                "[Epoch %d/%d] [Batch %d] [recon_pos:%.3f][reconneg:%.3f]"
                % (epoch, opt.n_epochs, i, recon_pos,recon_neg)
            )
   
   
    if((np.mean(dxx_list)<0.015 or np.mean(dxz_list)<0.015) and epoch>300):
        break
    eva_dic= test_eva(generator,encoder,discriminator,epoch,val_loader,test_loader,opt)
    auc_re = auc_re.append(eva_dic,ignore_index=True)
    end = time.time()
    time_epoch = end-start
    print(time_epoch)
    if(eva_dic['val_recon']>best_val_recon):
        best_test_recon = eva_dic['test_recon']
        best_val_recon = eva_dic['val_recon']
    
    if(eva_dic['val_zs']>best_val_zs):
        best_test_zs = eva_dic['test_zs']
        best_val_zs = eva_dic['val_zs']
    
    print(
                "[Epoch %d/%d] [val_recon:%.3f][test_recon:%.3f] [val_zs:%.3f][test_zs:%.3f] [best_recon:%.3f][best_zs:%.3f][epoch_time:%.3f]"
                % (epoch, opt.n_epochs,eva_dic['val_recon'],eva_dic['test_recon'],eva_dic['val_zs'],eva_dic['test_zs'],best_test_recon,best_test_zs,time_epoch)
            )