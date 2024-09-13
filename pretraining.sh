### CUB + ResNet18

python train.py \
--save-dir CUB/resnet18 \
--gpu 0 \
--model resnet18 \
--max-lr 0.05 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB

python train.py \
--pretrained-net ./pretrained_weight/resnet18_inet_torch.pth \
--save-dir CUB/resnet18_inet \
--gpu 0 \
--model resnet18 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB

### CUB + ResNet50

python train.py \
--save-dir CUB/resnet50 \
--gpu 0 \
--model resnet50 \
--max-lr 0.05 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB

python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CUB/resnet50_inet \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB

python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_moco_v2_800ep.pth \
--save-dir CUB/resnet50_moco \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB


### CARS + ResNet18

python train.py \
--save-dir CARS/resnet18 \
--gpu 0 \
--model resnet18 \
--max-lr 0.05 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS

python train.py \
--pretrained-net ./pretrained_weight/resnet18_inet_torch.pth \
--save-dir CARS/resnet18_inet \
--gpu 0 \
--model resnet18 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS

### CARS + ResNet50

python train.py \
--save-dir CARS/resnet50 \
--gpu 0 \
--model resnet50 \
--max-lr 0.05 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS

python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CARS/resnet50_inet \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS

python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_moco_v2_800ep.pth \
--save-dir CARS/resnet50_moco \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS