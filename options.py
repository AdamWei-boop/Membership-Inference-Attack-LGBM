#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():

    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: N")
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--num_items_train', type=int, default=500, help="dataset size for each user")
    parser.add_argument('--num_items_test', type=int, default=100, help="dataset size for each user")       
    parser.add_argument('--local_bs', type=int, default=50, help="local batch size: B")
    parser.add_argument('--lr_orig', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_g_orig', type=float, default=0.005, help='global learning rate')
    parser.add_argument('--beta_1', type=float, default=0.9, help='momentum parameter')
    parser.add_argument('--beta_2', type=float, default=0.99, help='momentum parameter')    
    parser.add_argument('--momentum', type=float, default=0.0, help='SGD momentum (default: 0.9)')
    parser.add_argument('--degree_noniid', type=float, default=0, help='degree of non-i.i.d.')
    parser.add_argument('--iid', type=bool, default=True, help='whether i.i.d. or not')
    parser.add_argument('--ratio_train', type=list, default=[1], help="distribution of training datasets")
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--batch_type', type=str, default='BSGD', help="BSGD or mini-BSGD")
    parser.add_argument('--set_algo_type', type=list, default=['Fed-SPA'], help="non-compress or Fed-SPA or Fed-DPMEV, the selected algorithm")
    parser.add_argument('--acceleration', type=bool, default=True)

    # differential privacy
    parser.add_argument('--DP', type=bool, default=True, help='whether differential privacy or not')  
    parser.add_argument('--delta', type=float, default=1e-3, help='The parameter of DP') 
    
    #sparse setting
    parser.add_argument('--set_p', type=list, default=[1], help='The list of sparse rate')  
    
    # other arguments
    parser.add_argument('--num_experiments', type=int, default=1, help="number of experiments")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=-1)
    
    # key setting    
    parser.add_argument('--privacy_budget', type=int, default=10, help='The value of epsilon')
    parser.add_argument('--set_privacy_budget', type=list, default=[2,5,10,20], help='The list of epsilon')
    parser.add_argument('--clipthr', type=int, default=5, help='The clipping threshold')
    parser.add_argument('--set_clipthr', type=list, default=[20], help='The set of clipping thresholds')
    parser.add_argument('--set_noise_scale', type=list, default=[], help='The noise scale list') 
    parser.add_argument('--lr_decay', default=True, help="Learning rate decay")
    parser.add_argument('--aggregation_mask', default=True)
    
    args = parser.parse_args()
    
    if args.dataset == 'mnist':
        args.set_num_users = [5]
        args.set_epochs = [25]
        args.local_ep = 300
        args.local_bs = 10
        args.lr_orig = 0.01
        args.lr_g_orig = 0.01
    else:    
        args.set_num_users = [10,20,30,50]
        args.set_epochs = [150, 200, 250, 300]
        
    return args