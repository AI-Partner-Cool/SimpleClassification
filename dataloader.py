#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import torchvision 
import torch
import torchvision.transforms as transforms 

def Trainloader_cifar10(batch_size, train_dir) : 
	
    ## train transform
    transform_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ## train set image statistic
                                        ])
    
    trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last = True) ## workers can surely be optimized...
	
    return trainloader

def Testloader_cifar10(batch_size, test_dir) : 

	## test transform
    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
    testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last = False)
	
	
    return testloader