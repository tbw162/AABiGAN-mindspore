# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 13:40:09 2023

@author: lab503
"""

import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import numpy as np
import mindspore as ms
import mindspore.dataset.vision as vision
def create_loader(opt):
    train_data = np.load('cifar_train_data.npy')
    train_label = np.load('cifar_train_label.npy')
    if(opt.gamma_p==0):
        data1 = train_data[train_label==opt.normal_digit]
        target1 = train_label[train_label==opt.normal_digit]
    else:
        data1_p = train_data[train_label==opt.normal_digit]
        data1_n = train_data[train_label!=opt.normal_digit]
        target1_p = train_label[train_label==opt.normal_digit]
        target1_n = train_label[train_label!=opt.normal_digit]
        randIdx = np.arange(data1_n.shape[0])
        np.random.shuffle(randIdx)
        normal_num = data1_p.shape[0]
        abnormal_num = int((normal_num*opt.gamma_p)/(1-opt.gamma_p))
        data1 = np.concatenate((data1_p,data1_n[randIdx[:abnormal_num]]),axis=0)
        target1 = np.concatenate((target1_p,target1_n[randIdx[:abnormal_num]]),axis=0)
        
    print(data1.shape)
    print(target1.shape)
    train_pos = ds.NumpySlicesDataset((data1, target1), ["data", "label"],shuffle=True)
    #train_pos = train_pos.map(operations=vision.Resize(size=(32, 32)), input_columns="data")
    train_pos = train_pos.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    train_pos = train_pos.batch(batch_size=opt.batch_size,drop_remainder=False)
    
    
    
    if(opt.gamma_p==0):
        data2 = train_data[train_label!=opt.normal_digit]
        target2 = train_label[train_label!=opt.normal_digit]
    else:
        data2 = data1_n[randIdx[abnormal_num:]]
        target2 = target1_n[randIdx[abnormal_num:]]
        
    
    if(opt.k==1):
       
        data2 = data2[target2==opt.auxiliary_digit]
       
        target2 = target2[target2==opt.auxiliary_digit]
        
        
    else:
        anomaly_list = list(np.arange(0,10))
        anomaly_list.remove(opt.normal_digit)
        randIdx_list = np.arange(len(anomaly_list))
        np.random.shuffle(randIdx_list)
        if(opt.k==2):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])] 
        elif(opt.k==3):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]]) |(target2==anomaly_list[randIdx_list[2]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]])] 
        elif(opt.k==5):
            data2 = data2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])]
            target2 = target2[(target2==anomaly_list[randIdx_list[0]]) |(target2==anomaly_list[randIdx_list[1]])|(target2==anomaly_list[randIdx_list[2]]) |(target2==anomaly_list[randIdx_list[3]]) |(target2==anomaly_list[randIdx_list[4]])] 
        else:
            data2 = data2
            target2 = target2
    randIdx = np.arange(data2.shape[0])
    np.random.shuffle(randIdx)
    unlabeled_num = data1.shape[0]
    auxiliary_num = int((unlabeled_num*opt.gamma_l)/(1-opt.gamma_l))
  

    data2 = data2[randIdx[:auxiliary_num]]
    target2 = np.array(target2)[randIdx[:auxiliary_num]]

    train_neg = ds.NumpySlicesDataset((data2, target2), ["data", "label"],shuffle=True)
    #train_neg = train_neg.map(operations=vision.Resize(size=(32, 32)), input_columns="data")
    train_neg = train_neg.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    train_neg = train_neg.batch(batch_size=opt.batch_size//9,drop_remainder=False)
    
    test_data = np.load('cifar_test_data.npy')
    test_label = np.load('cifar_test_label.npy')
    test_data_normal = test_data[test_label==opt.normal_digit]
    test_label_normal = test_label[test_label==opt.normal_digit]
    test_data_abnormal = test_data[test_label!=opt.normal_digit]
    test_label_abnormal = test_label[test_label!=opt.normal_digit]
    randIdx_test_normal = np.arange(test_data_normal.shape[0])
    randIdx_test_abnormal = np.arange(test_data_abnormal.shape[0])
    np.random.shuffle(randIdx_test_normal)
    np.random.shuffle(randIdx_test_abnormal)
    val_data = np.concatenate((test_data_normal[randIdx_test_normal[:200]],test_data_abnormal[randIdx_test_abnormal[:1800]]),axis=0)
    val_label = np.concatenate((test_label_normal[randIdx_test_normal[:200]],test_label_abnormal[randIdx_test_abnormal[:1800]]),axis=0)
    val_loader = ds.NumpySlicesDataset((val_data,val_label),['data','label'])
    #val_loader = val_loader.map(operations=vision.Resize(size=(32, 32)), input_columns="data")
    val_loader = val_loader.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    val_loader = val_loader.batch(batch_size=opt.batch_size,drop_remainder=False)
    test_data = np.concatenate((test_data_normal[randIdx_test_normal[200:]],test_data_abnormal[randIdx_test_abnormal[1800:]]),axis=0)
    test_label = np.concatenate((test_label_normal[randIdx_test_normal[200:]],test_label_abnormal[randIdx_test_abnormal[1800:]]),axis=0)
    test_loader = ds.NumpySlicesDataset((test_data,test_label),['data','label'])
    #test_loader = test_loader.map(operations=vision.Resize(size=(32, 32)), input_columns="data")
    test_loader = test_loader.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    test_loader = test_loader.batch(batch_size=opt.batch_size,drop_remainder=False)
    
    return train_pos, train_neg, val_loader, test_loader