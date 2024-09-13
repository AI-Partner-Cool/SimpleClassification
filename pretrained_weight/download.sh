wget https://download.pytorch.org/models/resnet18-f37072fd.pth
mv resnet18-f37072fd.pth resnet18_inet_torch.pth


# https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/
# resnet50 with sota tricks, 80.858 on Inet
# Mainly from "ResNet strikes back: An improved training procedure in timm", Ross Wightman, Hugo Touvron, Hervé Jégou, 2021 
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mv resnet50-11ad3fa6.pth resnet50_inet_torch_sota.pth 


# https://github.com/facebookresearch/moco?tab=readme-ov-file#models
# resnet50 with MOCOV2 800 Epochs (Self-supervised representation learning)
# from "Improved Baselines with Momentum Contrastive Learning", Xinlei Chen, Haoqi Fan, Ross Girshick, Kaiming He, 2020 
wget https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
mv moco_v2_800ep_pretrain.pth.tar resnet50_inet_moco_v2_800ep.pth 

