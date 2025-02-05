#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : AI Partner
# @Email   : ai.partner.cool@outlook.com

import utils.option
import os
args = utils.option.get_args_parser()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


 
import json 
import datetime
import resource
import dataloader


import torch.backends.cudnn
import torch.utils.tensorboard

import model.resnet_cifar
import model.resnet
import model.deit
import model.convnext




import test
import utils.utils
import utils.lr_scheduler

import utils.ema

def get_stats():
    stats = {'train_ce_loss': utils.utils.AverageMeter(),
            'train_acc': utils.utils.AverageMeter(),
            'test_acc': utils.utils.AverageMeter(),
            'test_acc_ema': utils.utils.AverageMeter(),
            'test_aurc': utils.utils.AverageMeter(),
            'test_aurc_ema': utils.utils.AverageMeter(),
            'lr' : utils.utils.AverageMeter()}
    return stats

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

torch.backends.cudnn.benchmark = True

## tensorboard and logger
writer = torch.utils.tensorboard.SummaryWriter(args.save_dir)
logger = utils.utils.get_logger(args.save_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
logger.info('Log saved in {}'.format(args.save_dir))

## define dataloader
if args.dataset == 'CIFAR10':
    train_loader = dataloader.Trainloader_cifar10(args.batch_size, args.train_dir, args.train_size)
    test_loader = dataloader.Testloader_cifar10(args.batch_size, args.test_dir, args.test_size)

elif args.dataset == 'CIFAR100':
    train_loader = dataloader.Trainloader_cifar100(args.batch_size, args.train_dir, args.train_size)
    test_loader = dataloader.Testloader_cifar100(args.batch_size, args.test_dir, args.test_size)

elif args.dataset in ['CUB', 'CARS']:
    train_loader = dataloader.Trainloader_ImageFolder(args.batch_size, args.train_dir, args.train_size)
    test_loader = dataloader.Testloader_ImageFolder(args.batch_size, args.test_dir, args.test_size)

iter_per_epoch = len(train_loader)

## define model
if args.model == 'resnet18_cifar' : 
    net = model.resnet_cifar.ResNet18(args.nb_cls)
elif args.model == 'resnet18' :
    net = model.resnet.ResNet18(args.nb_cls)
    if args.pretrained_net is not None : 
        net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)
elif args.model == 'resnet50' :
    net = model.resnet.ResNet50(args.nb_cls)
    if args.pretrained_net is not None : 
        net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)
elif args.model == 'deit_base_patch16_384' :
    net = model.deit.deit_base_patch16_384(args.nb_cls)
    if args.pretrained_net is not None : 
        net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)
elif args.model == 'convnext_base' :
    net = model.convnext.convnext_base(args.nb_cls)
    if args.pretrained_net is not None : 
        net = utils.utils.load_pretrained_net(net, args.pretrained_net, logger)

net.cuda()
net_ema = utils.ema.ModelEMA(net)

## define optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=args.min_lr, momentum=0.9, weight_decay=args.weight_decay)

## define Warmup cos lr scheduler
lr_scheduler = utils.lr_scheduler.Warmup_cos_lr(args.max_lr, args.min_lr, iter_per_epoch, args.nb_epoch, args.warmup_epoch)

## define criterion
CE_Loss = torch.nn.CrossEntropyLoss()

## define stats
stats = get_stats()

best_acc, best_acc_ema, best_aurc, best_aurc_ema, iter_counter = 0, 0, 1000, 1000, 0

## Sacler: Gradient scaling helps prevent gradients with small magnitudes from flushing to zero (“underflowing”) when training with mixed precision.
scaler = torch.cuda.amp.GradScaler()

# start Train
for epoch in range(args.nb_epoch):
    net.train()
        
    for i, batch in enumerate(train_loader) : 
            
        lr = lr_scheduler.update_lr(iter_counter)
        stats['lr'] = lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        image, target = batch
        image, target = image.cuda(), target.cuda()
        
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            logits = net(image)
            loss = CE_Loss(logits, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        net_ema.update(net)
        prec, correct = utils.utils.accuracy(logits, target)
        stats['train_acc'].update(prec.item(), image.size(0))
        stats['train_ce_loss'].update(loss.item(), image.size(0))
        
        if i % 100 == 99 : 
            logger.info(f"{datetime.datetime.now()} \t LR {stats['lr']:.5f} \t Epoch {epoch} \t Batch {i} \t Train Acc. {stats['train_acc'].avg:.2%} \t Train Loss {stats['train_ce_loss'].avg:.2f}")
        iter_counter += 1
    
    net.eval()
    stats['test_acc'], stats['test_aurc'], _, _ = test.test(test_loader, net)
    stats['test_acc_ema'], stats['test_aurc_ema'], _, _ = test.test(test_loader, net_ema.ema)
    msg = f"TEST: {datetime.datetime.now()} \t \
            Epoch {epoch} \t \
            Train Acc. {stats['train_acc'].avg:.2%} \t \
            Test Acc. {stats['test_acc'].avg:.2%} (Prev. Best {best_acc:.2%}) \t \
            Test AURC {stats['test_aurc']:.2f} (Prev. Best {best_aurc:.2f}) \t \n \
            EMA Test Acc. {stats['test_acc_ema'].avg:.2%} (EMA Prev. Best {best_acc_ema:.2%}) \t \
            EMA Test AURC {stats['test_aurc_ema']:.2f} (EMA Prev. Best {best_aurc_ema:.2f}) \t \
            Train Loss {stats['train_ce_loss'].avg:.2f}"
    logger.info(msg)
    
    for metric in stats:
        writer.add_scalar(metric, stats[metric].avg if metric not in  ['lr', 'test_aurc', 'test_aurc_ema'] else stats[metric], epoch)
    
    if stats['test_acc'].avg > best_acc :
        logger.info(f"Accuracy improved from {best_acc:.2%} to {stats['test_acc'].avg:.2%}!!!")
        best_acc = stats['test_acc'].avg
        torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_acc_net.pth'))
    
    if stats['test_acc_ema'].avg > best_acc_ema :
        logger.info(f"EMA Accuracy improved from {best_acc_ema:.2%} to {stats['test_acc_ema'].avg:.2%}!!!")
        best_acc_ema = stats['test_acc_ema'].avg
        torch.save(net_ema.ema.state_dict(), os.path.join(args.save_dir, 'best_acc_net_ema.pth'))

    if stats['test_aurc'] < best_aurc :
        logger.info(f"AURC improved from {best_aurc:.2f} to {stats['test_aurc']:.2f}!!!")
        best_aurc = stats['test_aurc']
        torch.save(net.state_dict(), os.path.join(args.save_dir, 'best_aurc_net.pth'))

    if stats['test_aurc_ema'] < best_aurc_ema :
        logger.info(f"EMA AURC improved from {best_aurc_ema:.2f} to {stats['test_aurc_ema']:.2f}!!!")
        best_aurc_ema = stats['test_aurc_ema']
        torch.save(net_ema.ema.state_dict(), os.path.join(args.save_dir, 'best_aurc_net_ema.pth'))
    
    ## re-initialize stats
    stats = get_stats()