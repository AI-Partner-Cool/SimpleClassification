#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import math
import os
from functools import partial

def Warm_cos_lr(max_lr,
                min_lr,
                total_iter,
                warmup_total_iter,
                iter):

    """Cosine learning rate with warm up."""
    
    if iter <= warmup_total_iter:
        lr = max_lr * pow(iter / float(warmup_total_iter), 2)
    
    else:
        lr = min_lr + 0.5 * (max_lr - min_lr) * (
            1.0
            + math.cos(
                math.pi
                * (iter - warmup_total_iter)
                / (total_iter - warmup_total_iter)
            )
        )
    return lr




class Warmup_cos_lr:
    def __init__(self,
                 max_lr,
                 min_lr,
                 iter_per_epoch,
                 num_epoch,
                 warmup_epoch):
        """
        
        Args:
            max_lr : maximun learning rate in the cosine learning rate scheduler
            min_lr : minimum learning rate in the cosine learning rate scheduler (used in no aug epochs)
            iter_per_epoch : number of iterations in one epoch.
            
            num_epoch : number of epochs in training.
            warmup_epoch : number of epochs in warm-up.
        """

        total_iter = iter_per_epoch * num_epoch
        warmup_iter = iter_per_epoch * warmup_epoch
        
        
        self.lr_func = partial(Warm_cos_lr,
                               max_lr,
                               min_lr,
                               total_iter,
                               warmup_iter)

    def update_lr(self, iter):
        return self.lr_func(iter)

    