#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com
import os 

import model.resnet_cifar
import model.resnet
import model.deit

import dataloader
import utils.utils
import utils.option
import json 

import torch 
import numpy as np
from sklearn.metrics import auc


@torch.no_grad()
def compute_aurc(confidences, predictions, targets):
    """Compute Area Under Risk-Coverage Curve"""
    n_samples = len(confidences)
    
    # Sort by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_confidences = confidences[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]
    
    # Calculate risk and coverage
    risks = []
    coverages = []
    correct = sorted_predictions == sorted_targets
    
    for i in range(n_samples):
        coverage = (i + 1) / n_samples
        selected_correct = correct[:i+1]
        risk = 1.0 - np.mean(selected_correct) if len(selected_correct) > 0 else 0.0
        
        risks.append(risk)
        coverages.append(coverage)
    
    # Compute AURC using sklearn's auc function
    aurc = auc(coverages, risks)
    return aurc, risks, coverages

@torch.no_grad()
def test(test_loader, net):
    test_acc = utils.utils.AverageMeter()
    all_confidences = []
    all_predictions = []
    all_targets = []
    
    for i, batch in enumerate(test_loader):
        image, target = batch
        image, target = image.cuda(), target.cuda()
        logits = net(image)
        
        # Calculate accuracy
        prec, correct = utils.utils.accuracy(logits, target)
        test_acc.update(prec.item(), image.size(0))
        
        # Store predictions and confidences
        softmax_probs = torch.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmax_probs, dim=1)
        
        all_confidences.extend(confidences.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    # Convert to numpy arrays
    all_confidences = np.array(all_confidences)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute AURC
    aurc, risks, coverages = compute_aurc(all_confidences, all_predictions, all_targets)
    
    return test_acc, aurc * 1000, risks, coverages

if __name__ == "__main__": 

    args = utils.option.get_args_parser()
    logger = utils.utils.get_logger(args.save_dir, name="test.log")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info('Log saved in {}'.format(os.path.join(args.save_dir, "test.log")))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ## define dataloader
    if args.dataset == 'CIFAR10':
        test_loader = dataloader.Testloader_cifar10(args.batch_size, args.test_dir, args.test_size)

    elif args.dataset == 'CIFAR100':
        test_loader = dataloader.Testloader_cifar100(args.batch_size, args.test_dir, args.test_size)

    elif args.dataset in ['CUB', 'CARS']:
        test_loader = dataloader.Testloader_ImageFolder(args.batch_size, args.test_dir, args.test_size)

    ## define model
    if args.model == 'resnet18_cifar' : 
        net = model.resnet_cifar.ResNet18(args.nb_cls)
    elif args.model == 'resnet18' :
        net = model.resnet.ResNet18(args.nb_cls)
    elif args.model == 'resnet50' :
        net = model.resnet.ResNet50(args.nb_cls)
    elif args.model == 'deit_base_patch16_384' :
        net = model.deit.deit_base_patch16_384(args.nb_cls)
    
    net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)
    net.cuda()
    net.eval()
    test_acc, aurc, risks, coverages = test(test_loader, net)
    
    msg = 'Dataset {} \t Test Acc. {:.2%} \t AURC {:.2f}'.format(
        args.dataset, test_acc.avg, aurc
    )
    logger.info(msg) 

    