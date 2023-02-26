# -*- coding: utf-8 -*-


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
import mindspore.ops as ops



def test_eva(G,E,D,epoch,val_loader,test_loader,opt):
    #data_path=data_path=PACK_PATH+"/normal_test"
    #test_loader = load_dataset(256, data_path, 1)
    PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    G.set_train(False)
    E.set_train(False)
    D.set_train(False)
    
    target_all_val = []
    rec_all_val = []
    z_score_val = []
    z_score3_val = []
   
    target_all_test = []
    rec_all_test = []
    z_score_test = []
    z_score3_test = []
    
        
    for idx, (image, target) in enumerate(val_loader):
        #print(target)
        target_all_val.append(target.asnumpy())
        #print(target_all_val)
        score1= ((G(E(image))-image)**2).sum(axis=(1,2,3))
        rec_all_val.append(score1.asnumpy())
        
        score3 = ops.abs(D(image,E(image))[0] - D(G(E(image)),E(image))[0])
        z_score3_val.append(score3.asnumpy())
        
        score4 = (E(image)**2).sum(axis=1)
        z_score_val.append(score4.asnumpy())
            
    target_all_val = np.concatenate(target_all_val,axis=0)
    rec_all_val = np.concatenate(rec_all_val,axis=0)
    z_score_val = np.concatenate(z_score_val,axis=0)
    z_score3_val = np.concatenate(z_score3_val,axis=0)
    
    
      
    
    gt_val = (target_all_val == opt.normal_digit).astype(int)
    auc_recon_val = roc_auc_score(gt_val,-1*rec_all_val) 
    auc_score_val = roc_auc_score(gt_val,-1*z_score_val)
    auc_score3_val = roc_auc_score(gt_val,-1*z_score3_val)
    
    target_all_test = []
    print('test')
    #print(target_all_test)
    for idx, (image, target) in enumerate(test_loader):
        #print(target)
        #print(target_all_test)
        target_all_test.append(target.asnumpy())
        
        score1= ((G(E(image))-image)**2).sum(axis=(1,2,3))
        rec_all_test.append(score1.asnumpy())
        
        score3 = ops.abs(D(image,E(image))[0] - D(G(E(image)),E(image))[0])
        z_score3_test.append(score3.asnumpy())
        
        score4 = (E(image)**2).sum(axis=1)
        z_score_test.append(score4.asnumpy())
                
    target_all_test = np.concatenate(target_all_test,axis=0)
    rec_all_test = np.concatenate(rec_all_test,axis=0)
    z_score_test = np.concatenate(z_score_test,axis=0)
    z_score3_test = np.concatenate(z_score3_test,axis=0)
    
    gt_test = (target_all_test == opt.normal_digit).astype(int)
    auc_recon_test = roc_auc_score(gt_test,-1*rec_all_test)
    auc_score_test = roc_auc_score(gt_test,-1*z_score_test)
    auc_score3_test = roc_auc_score(gt_test,-1*z_score3_test)
    
    
    eva_dic = {}
    eva_dic['val_recon'] = auc_recon_val
    eva_dic['val_zs'] = auc_score_val
    eva_dic['val_zs3'] = auc_score3_val
    eva_dic['test_recon'] = auc_recon_test
    eva_dic['test_zs'] = auc_score_test
    eva_dic['test_zs3'] = auc_score3_test
    eva_dic['epoch'] = epoch
    return eva_dic
