#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='A simple code for image classification',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## save dir, training gpu
    parser.add_argument('--save-dir', default='./out_cifar10', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    ## model 
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18_cifar', 'resnet18', 'resnet50'], help='which model?')
    parser.add_argument('--pretrained-net', default=None, type=str, help='if use pretrained model, set pretrained model path')
    
    
    ## optimization
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    parser.add_argument('--nb-epoch', default=200, type=int, help='Total number of training epochs ')
    parser.add_argument('--warmup-epoch', default=5, type=int, help='Warmup epochs')
    parser.add_argument('--max-lr', default=0.1, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--min-lr', default=1e-4, type=float, help='Min learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')

    parser.add_argument("--train-size", type=int, default=224, help="image train resolution")
    parser.add_argument("--test-size", type=int, default=224, help="image test resolution")
    
    ## dataset setting: support CIFAR10, CIFAR100, CUB and CARS
    subparsers = parser.add_subparsers(title="dataset setting", dest="dataset")
    # --- CIFAR10 --- #
    CIFAR10 = subparsers.add_parser("CIFAR10",
                                    description='Dataset parser for training on CIFAR10',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on CIFAR10")
    CIFAR10.add_argument("--train-dir", type=str, default='./data/CIFAR10', help="CIFAR10 train directory")
    CIFAR10.add_argument("--test-dir", type=str, default='./data/CIFAR10', help="CIFAR10 test directory")
    CIFAR10.add_argument("--nb-cls", type=int, default=10, help="number of classes in CIFAR10")
    
    # --- CIFAR100 --- #
    CIFAR100 = subparsers.add_parser("CIFAR100",
                                    description='Dataset parser for training on CIFAR100',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on CIFAR100")
    CIFAR100.add_argument("--train-dir", type=str, default='./data/CIFAR100', help="CIFAR100 train directory")
    CIFAR100.add_argument("--test-dir", type=str, default='./data/CIFAR100', help="CIFAR100 test directory")
    CIFAR100.add_argument("--nb-cls", type=int, default=100, help="number of classes in CIFAR100")
    
    # --- CUB --- #
    CUB = subparsers.add_parser("CUB",
                                    description='Dataset parser for training on CUB',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on CUB")
    CUB.add_argument("--train-dir", type=str, default='./data/CUB_200_2011/train', help="CUB train directory")
    CUB.add_argument("--test-dir", type=str, default='./data/CUB_200_2011/test', help="CUB test directory")
    CUB.add_argument("--nb-cls", type=int, default=200, help="number of classes in CUB")

    # --- CARS --- #
    CARS = subparsers.add_parser("CARS",
                                    description='Dataset parser for training on Stanford CARS',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on Stanford CARS")
    CARS.add_argument("--train-dir", type=str, default='./data/Stanford_CARS/train', help="CARS train directory")
    CARS.add_argument("--test-dir", type=str, default='./data/Stanford_CARS/test', help="CARS test directory")
    CARS.add_argument("--nb-cls", type=int, default=196, help="number of classes in CARS")
    return parser.parse_args()