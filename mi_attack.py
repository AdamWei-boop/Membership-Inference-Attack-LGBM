# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:25:45 2022

@author: weikang
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import lightgbm as lgb
import datetime
import argparse

from dataloaders import *
from torchvision import datasets, transforms
from model import *
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from trainer import *
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score      
from data_target import prepare_target_data
from opacus import PrivacyEngine

import warnings

warnings.filterwarnings("ignore")


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        image, label = self.dataset[int(self.idxs[item])]
        return image, label
    
def split_dataset(dataset, target_size):
    
    #For simplicity we are only using orignal training set and splitting into 4 equal parts
    #and assign it to Target train/test and Shadow train/test.
    total_size = len(dataset)
    
    indices = list(range(total_size))
    
    np.random.shuffle(indices)
    
    #Target set and shadow set
    target_idxs = indices[:target_size]
    shadow_idxs = indices[target_size:]
    
    return target_idxs, shadow_idxs

def mi(args):
    
    print("Start {} datasets".format(args.dataset))
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.dataset == 'MNIST':
        
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                    ])
    
        data_train = datasets.MNIST(
                    './datasets/mnist/', 
                    train=True, 
                    download=True,
                    transform=transform)
        data_test = datasets.MNIST(
                    './datasets/mnist/', 
                    train=False, 
                    download=True,
                    transform=transform)
        
    elif args.dataset == 'CIFAR10':
    
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        data_train = datasets.CIFAR10(
                    './datasets/cifar/', 
                    train=True, 
                    transform=transform, 
                    target_transform=None, 
                    download=True)
        data_test = datasets.CIFAR10(
                    './datasets/cifar/', 
                    train=False, 
                    transform=transform, 
                    target_transform=None, 
                    download=True)

    if args.is_target_train:
        
        # Prepare the datasets
        train_target_idx, train_shadow_idx = split_dataset(data_train, args.train_target_size)
        test_target_idx, test_shadow_idx = split_dataset(data_test, args.test_target_size)
        
        data_train_target = DatasetSplit(data_train, train_target_idx)
        data_test_target = DatasetSplit(data_test, test_target_idx)
        data_train_shadow = DatasetSplit(data_train, train_shadow_idx)
        data_test_shadow = DatasetSplit(data_test, test_shadow_idx)
        

        train_loader_target = torch.utils.data.DataLoader(data_train_target, batch_size=args.batch_size, shuffle=True)
        test_loader_target = torch.utils.data.DataLoader(data_test_target, batch_size=args.batch_size, shuffle=True)
        dataloaders_target = {"train": train_loader_target, "val": test_loader_target}
        dataset_sizes_target = {"train": len(data_train_target), "val": len(data_test_target)}
        print("Taille dataset", dataset_sizes_target)


        # Train the target model #
        print("Start training target model")        
        if args.dataset == 'MNIST':
            model_target = Net_mnist().to(device)
        elif args.dataset == 'CIFAR10':
            model_target = Net_cifar10().to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model_target.parameters(), lr=args.learning_rate, momentum=args.momentum)            
            
        if args.is_DP:
            alpha_list = list(np.arange(1.01, 20.0, 0.02))
            sample_rate = args.batch_size/len(data_train_target)
            max_grad_norm = 2.0
            privacy_engine = PrivacyEngine(
                model_target,
                sample_rate=sample_rate,
                alphas=alpha_list,
                noise_multiplier=None,
                max_grad_norm=max_grad_norm,
                secure_rng = False,
                target_epsilon=args.eps,
                target_delta = 1e-3,
                epochs = args.epochs,
            )
            privacy_engine.attach(optimizer)


        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decrease_lr_factor, gamma=args.decrease_lr_every)
        model_target, best_acc_target, data_test_set, label_test_set, class_test_set = train_model(model_target, criterion, optimizer, exp_lr_scheduler,dataloaders_target,dataset_sizes_target,
                            num_epochs=args.epochs)
    else:
        # Load the datasets
        dict_train = torch.load(args.data_train_dir)
        dict_train_shadow = torch.load(args.data_train_shadow_dir)
        dict_test = torch.load(args.data_val_dir)
        dict_test_shadow = torch.load(args.data_val_shadow_dir)
        
        data_train_target = DatasetSplit(data_train, dict_train[args.idx])
        data_test_target = DatasetSplit(data_test, dict_test[args.idx])
        data_train_shadow = DatasetSplit(data_train, dict_train_shadow[args.idx])
        data_test_shadow = DatasetSplit(data_test, dict_test_shadow[args.idx])

        
        train_loader_target = torch.utils.data.DataLoader(data_train_target, batch_size=args.batch_size, shuffle=True)
        test_loader_target = torch.utils.data.DataLoader(data_test_target, batch_size=args.batch_size, shuffle=True)
        
        dataloaders_target = {"train": train_loader_target, "val": test_loader_target}
        dataset_sizes_target = {"train": len(data_train_target), "val": len(data_test_target)}
        
        
        # Load the target model #      
        if args.dataset == 'MNIST':
            model_target = CNNMnist().to(device)
        elif args.dataset == 'CIFAR10':
            model_target = Net_cifar10().to(device)
        
        model_target.load_state_dict(torch.load(args.model_target_dir))
        
        criterion = nn.CrossEntropyLoss()
        
        model_target, best_acc_target, data_test_set, label_test_set, class_test_set\
                = prepare_target_data(model_target, criterion, dataloaders_target)
    
    # Train shadow model #
                
    print("Start training shadow models")
    all_shadow_models = []
    all_dataloaders_shadow = []
    data_train_set = []
    label_train_set = []
    class_train_set = []
    for num_model_sahdow in range(args.number_shadow_model):
        criterion = nn.CrossEntropyLoss()
        
        train_loader_shadow = torch.utils.data.DataLoader(data_train_shadow, batch_size=args.batch_size, shuffle=True)
        test_loader_shadow = torch.utils.data.DataLoader(data_test_shadow, batch_size=args.batch_size, shuffle=True)
        dataloaders_shadow = {"train": train_loader_shadow, "val": test_loader_shadow}
        dataset_sizes_shadow = {"train": len(data_train_shadow), "val": len(data_test_shadow)}
        print("Taille dataset", dataset_sizes_shadow)
        
        if args.dataset == 'MNIST':
            model_shadow = Net_mnist().to(device)
        elif args.dataset == 'CIFAR10':
            model_shadow = Net_cifar10().to(device)
        
        optimizer = optim.SGD(model_shadow.parameters(), lr=args.learning_rate, momentum=args.momentum)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decrease_lr_factor, gamma=args.decrease_lr_every)
        model_shadow, best_acc_sh, data_train_set_unit, label_train_set_unit, class_train_set_unit = train_model(model_shadow, criterion, optimizer, exp_lr_scheduler,dataloaders_target,dataset_sizes_target,
                           num_epochs=args.epochs)
        data_train_set.append(data_train_set_unit)
        label_train_set.append(label_train_set_unit)
        class_train_set.append(class_train_set_unit)
        all_shadow_models.append(model_shadow)
        all_dataloaders_shadow.append(dataloaders_shadow)
        
    print("Start getting dataset for the attack model")
    
    data_train_set = np.concatenate(data_train_set)
    label_train_set = np.concatenate(label_train_set)
    class_train_set = np.concatenate(class_train_set)
    #data_test_set, label_test_set, class_test_set = get_data_for_final_eval([model_target], [dataloaders_target], device)
    #data_train_set, label_train_set, class_train_set = get_data_for_final_eval(all_shadow_models, all_dataloaders_shadow, device)
    data_train_set, label_train_set, class_train_set = shuffle(data_train_set, label_train_set, class_train_set, random_state=args.seed)
    data_test_set, label_test_set, class_test_set = shuffle(data_test_set, label_test_set, class_test_set, random_state=args.seed)
    print("Taille dataset train", len(label_train_set))
    print("Taille dataset test", len(label_test_set))
    
    
    print("Start fitting attack model")
    # LGBM model
    model = lgb.LGBMClassifier(objective='binary', reg_lambda=args.reg_lambd, n_estimators=args.n_estimators)
    model.fit(data_train_set, label_train_set)
    y_pred_lgbm = model.predict(data_test_set)
    precision_general, recall_general, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm, y_true=label_test_set, average = "macro")
    accuracy_general = accuracy_score(y_true=label_test_set, y_pred=y_pred_lgbm)
    precision_per_class, recall_per_class, accuracy_per_class = [], [], []
    for idx_class, classe in enumerate(data_train.classes):
        all_index_class = np.where(class_test_set == idx_class)
        precision, recall, _, _ = precision_recall_fscore_support(y_pred=y_pred_lgbm[all_index_class], y_true=label_test_set[all_index_class], average = "macro")
        accuracy = accuracy_score(y_true=label_test_set[all_index_class], y_pred=y_pred_lgbm[all_index_class])
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        accuracy_per_class.append(accuracy)
    print("Inference attack end")
    return (precision_general, recall_general, accuracy_general, precision_per_class,\
            recall_per_class, accuracy_per_class)

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    
    # inference type
    parser.add_argument('--is_target_train', type=bool, default=True, help="Whether training target model")    
    parser.add_argument('--is_DP', type=bool, default=True, help="Whether training target model with DP")    
    
    # path:
    parser.add_argument('--model_target_dir', type=str, default='.')
    parser.add_argument('--data_train_dir', type=str, default='.')
    parser.add_argument('--data_val_dir', type=str, default='.')
    parser.add_argument('--data_train_shadow_dir', type=str, default='.')
    parser.add_argument('--data_val_shadow_dir', type=str, default='.')            
    
    # general:
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_target_size', type=int, default=2500, help='2 500, 5 000, 10 000, 15 000 // 4 600, 10 520, 19 920, 29 540')
    parser.add_argument('--test_target_size', type=int, default=1000)
    parser.add_argument('--number_shadow_model', type=int, default=25, help='25 50 MNIST and 100 cifar')
    
    # learning:
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--decrease_lr_factor', type=float, default=1e-7)
    parser.add_argument('--decrease_lr_every', type=int, default=1)
    
    parser.add_argument('--reg_lambd', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=10000)
        
    # statistics:
    parser.add_argument('--dataset', type=str, default='MNIST', help="MNIST, CIFAR10, CIFAR100")
    parser.add_argument('--training_size_value', type=list, default=[4600, 10520, 19920, 29540], help="[2500, 5000, 10000, 15000]")
    parser.add_argument('--number_shadow_value', type=list, default=[100], help='[2, 10 ,20, 50, 100]')
    parser.add_argument('--epoch_value', type=list, default=[100], help="[2, 10 ,20, 50, 100]")
    parser.add_argument('--experments', type=int, default=3)
        
    # differential privacy
    parser.add_argument('--eps', type=int, default=2)    
    parser.add_argument('--set_eps', type=list, default=[1,20,100])    
    
    args = parser.parse_args()
    
    
    # Example 1: training target model with various DP levels
    if args.is_target_train:
        
        acc_eps = []
        
        for args.eps in args.set_eps:
            acc_list = []
            for i in range(args.experments):
            
                precision_general, recall_general, accuracy_general, precision_per_class,\
                    recall_per_class, accuracy_per_class = mi(args)
                acc_list.append(accuracy_general)
                print('\nEps:', args.eps, 'Attack acc:', accuracy_general)    
            acc_eps.append(sum(acc_list)/len(acc_list))
        print('\nAttack acc list:', acc_eps)
        
    # Example 2: attack the trained model with allocated datasets   
    else:
    
        set_idx = [0,0,0,0]
        set_eps = [2,5,10,20]
        
        acc_eps = []
    
        for i in range(len(set_idx)):
            args.model_target_dir = './mi_attack/model-client{}_eps{}_delta0.001.ckpt'.format(set_idx[i], set_eps[i])
            args.data_train_dir = './mi_attack/dict_train-mnist.pt'
            args.data_val_dir = './mi_attack/dict_test-mnist.pt'
            args.data_train_shadow_dir = './mi_attack/dict_train_shadow-mnist.pt'
            args.data_val_shadow_dir = './mi_attack/dict_test_shadow-mnist.pt'
        
            acc_list = []
            for k in range(args.experments):
                precision_general, recall_general, accuracy_general, precision_per_class,\
                    recall_per_class, accuracy_per_class = mi(args)
                acc_list.append(accuracy_general)
                print('\nEps:', set_eps[i],'Accuracy:', accuracy_general)
            acc_eps.append(sum(acc_list)/len(acc_list))
        print('\nAttack acc list:', acc_eps)