#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import torchvision 
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

####### ---- CIFAR10 ---- #######

def Trainloader_cifar10(batch_size, train_dir) : 
	
    ## train transform
    transform_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ## train set image statistic
                                        ])
    
    trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
                             trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last = True
                             ) ## workers can surely be optimized...
	
    return trainloader

def Testloader_cifar10(batch_size, test_dir) : 

	## test transform
    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                    ])
    testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(
                            testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            drop_last = False
                            )
	
	
    return testloader


####### ---- CIFAR100 ---- #######
def Trainloader_cifar100(batch_size, train_dir) : 
	
    ## train transform
    transform_train = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ## train set image statistic
                                        ])
    
    trainset = torchvision.datasets.CIFAR100(root=train_dir, train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
                             trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             drop_last = True
                             ) ## workers can surely be optimized...
	
    return trainloader

def Testloader_cifar100(batch_size, test_dir) : 

	## test transform
    transform_test = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ## train set image statistic
                                    ])
    testset = torchvision.datasets.CIFAR100(root=test_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(
                            testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            drop_last = False
                            )
	
	
    return testloader


####### ---- StanfordCars ---- #######
def Trainloader_ImageFolder(batch_size, train_dir, train_size) : 
    ## imagenet mean + std
    transform_train = torchvision.transforms.Compose([
                                                    transforms.RandomResizedCrop(train_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])
    
    trainloader = DataLoader(
                            ImageFolder(train_dir, transform_train), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=2, 
                            drop_last=True
                            )
	
    return trainloader

def Testloader_ImageFolder(batch_size, test_dir, test_size) : 
    ## imagenet mean + std
    transform_test = torchvision.transforms.Compose([
                                                    transforms.Resize(int(test_size * 1.14)),
                                                    transforms.CenterCrop(test_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                    ])
    
    testloader = DataLoader(
                            ImageFolder(test_dir, transform_test),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2,
                            drop_last=False
                            )
	
    return testloader