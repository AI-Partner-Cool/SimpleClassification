#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import os
import sys
import logging
import numpy as np
import torch 

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(save_dir, name="run.log"):

    logger = logging.getLogger('Exp')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(save_dir, name)
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res[0], correct.squeeze()

def load_pretrained_net(net, pretrained_net, logger):
    
    logger.info('Loading pretrained weight from {}...'.format(pretrained_net))
    current_net_dict, pretrained_net_dict = net.state_dict(), torch.load(pretrained_net)

    ## Fix WEIGHT shape mismatch
    new_state_dict = {}
    for key in current_net_dict.keys() : 
        
        if key in pretrained_net_dict and current_net_dict[key].size() == pretrained_net_dict[key].size() : 
            new_state_dict[key] = pretrained_net_dict[key]
        else : 
            logger.info('{} missing in the pretrained weight...'.format(key))
            new_state_dict[key] = current_net_dict[key]
    
    ## Fix KEY mismatch with "strict=False", only load matched key
    ## "strict=False" is dangeous, might lead to bad initialization but w.o any errors
    ## "size mismatch for fc.weight: copying a param with shape torch.Size([1000, 512]) from checkpoint, the shape in current model is torch.Size([200, 512]).""
    ## should try "strict=True" first, check whether the ignored layer is correct then switch to strict = False
    net.load_state_dict(new_state_dict, strict=False) 
    return net
