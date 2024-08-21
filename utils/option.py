#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='A simple code to train CIFAR10',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--nb-epoch', default=200, type=int, help='Total number of training epochs ')
    parser.add_argument('--warmup-epoch', default=5, type=int, help='Warmup epochs')
    
    parser.add_argument('--train-dir', default='./data/', type=str, help='train directory')
    parser.add_argument('--test-dir', default='./data/', type=str, help='test directory')
    parser.add_argument('--nb-cls', default=10, type=int, help='nb of classes')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size')
    
    ## model 
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18', 'preact_resnet18'], help='which model?')
    
    ## optimizer
    parser.add_argument('--max-lr', default=0.1, type=float, help='Max learning rate for cosine learning rate scheduler')
    parser.add_argument('--min-lr', default=1e-4, type=float, help='Min learning rate for cosine learning rate scheduler')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')

    parser.add_argument('--save-dir', default='./out_cifar10', type=str, help='Output directory')
    parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')

    return parser.parse_args()