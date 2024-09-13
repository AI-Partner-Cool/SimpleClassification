#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com
import os 

import model.resnet_cifar
import model.resnet

import dataloader
import utils.utils
import utils.option
import json 

import torch 
@torch.no_grad()
def test(test_loader, net) : 
    test_acc = utils.utils.AverageMeter()
    for i, batch in enumerate(test_loader) : 
        image, target = batch
        image, target = image.cuda(), target.cuda()
        logits = net(image)
        prec, correct = utils.utils.accuracy(logits, target)
        test_acc.update(prec.item(), image.size(0))
    return test_acc
    
if __name__ == "__main__": 

    args = utils.option.get_args_parser()
    logger = utils.utils.get_logger(args.save_dir, name="test.log")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    logger.info('Log saved in {}'.format(os.path.join(args.save_dir, "test.log")))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    ## define dataloader
    if args.dataset == 'CIFAR10':
        test_loader = dataloader.Testloader_cifar10(args.batch_size, args.test_dir)
    elif args.dataset == 'CIFAR100':
        test_loader = dataloader.Testloader_cifar100(args.batch_size, args.test_dir)
    elif args.dataset in ['CUB', 'CARS']:
        test_loader = dataloader.Testloader_ImageFolder(args.batch_size, args.test_dir, args.test_size)

    ## define model
    if args.model == 'resnet18_cifar' : 
        net = model.resnet_cifar.ResNet18(args.nb_cls)
    elif args.model == 'resnet18' :
        net = model.resnet.ResNet18(args.nb_cls)
    elif args.model == 'resnet50' :
        net = model.resnet.ResNet50(args.nb_cls)
    
    net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)
    net.cuda()
    net.eval()
    test_acc = test(test_loader, net)
    msg = 'Dataset {} \t Test Acc. {:.2%}'.format(args.dataset, test_acc.avg) 
    logger.info(msg) 

    