# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:37:21 2022

@author: weikang
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_target_data(model, criterion, dataloaders):

    since = time.time()


    X = []
    Y = []
    C = []

    retunr_value_train = np.zeros(4)

    for phase in ['train', 'val']:

        model.eval()   # Set model to evaluate mode
    
        running_loss = 0.0
        running_corrects = 0
    
        # Iterate over data.
        for batch_idx, (data, target) in enumerate(dataloaders[phase]):
            inputs, labels = data.to(device), target.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
    
            for out in outputs.cpu().detach().numpy():
                X.append(out)
                if phase == "train":
                    Y.append(1)
                else:
                    Y.append(0)
            for cla in labels.cpu().detach().numpy():
                C.append(cla)
    
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders)
        epoch_acc = running_corrects.double() / len(dataloaders)

        if phase == 'train':
            retunr_value_train[0] = epoch_loss
            retunr_value_train[1] = epoch_acc
        else:
            retunr_value_train[2] = epoch_loss
            retunr_value_train[3] = epoch_acc

        #print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("DONE TRAIN")

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model, retunr_value_train, np.array(X), np.array(Y), np.array(C)