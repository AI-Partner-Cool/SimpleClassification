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


### DEIT-B (DEIT-III + Inet21K)

python train.py \
--save-dir CARS/deit_base_patch16_384_Inet21K_BS32_epoch150_Lr2e3 \
--gpu 1 \
--model deit_base_patch16_384 \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 32 \
--pretrained-net ./pretrained_weight/deit_3_base_384_21k.pth \
--test-size 384 CARS

python train.py \
--save-dir CUB/deit_base_patch16_384_Inet21K_BS32_epoch150_Lr2e3 \
--gpu 6 \
--model deit_base_patch16_384 \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 32 \
--pretrained-net ./pretrained_weight/deit_3_base_384_21k.pth \
--test-size 384 CUB

python train.py \
--save-dir CIFAR10/deit_base_patch16_384_Inet21K_BS32_epoch150_Lr2e3 \
--gpu 7 \
--model deit_base_patch16_384 \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 32 \
--pretrained-net ./pretrained_weight/deit_3_base_384_21k.pth \
--test-size 384 CIFAR10

python train.py \
--save-dir CIFAR100/deit_base_patch16_384_Inet21K_BS32_epoch150_Lr2e3 \
--gpu 8 \
--model deit_base_patch16_384 \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 32 \
--pretrained-net ./pretrained_weight/deit_3_base_384_21k.pth \
--test-size 384 CIFAR100


### ConvNext (ConvNext-B + Inet22K-1K + 384 * 384)
#### CIFAR10 
python train.py \
--save-dir CIFAR10/convnext_base_384_Inet22K_1K_BS48_epoch50_lr002 \
--gpu 4 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_1k_384.pth \
--test-size 384 CIFAR10

#### CIFAR100
python train.py \
--save-dir CIFAR100/convnext_base_384_Inet22K_1K_BS48_epoch50_lr002 \
--gpu 4 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_1k_384.pth \
--test-size 384 CIFAR100

### CUB 
python train.py \
--save-dir CUB/convnext_base_384_Inet22K_1K_BS48_epoch150_lr002 \
--gpu 4 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_1k_384.pth \
--test-size 384 CUB

### CARS
python train.py \
--save-dir CARS/convnext_base_384_Inet22K_1K_BS48_epoch150_lr002 \
--gpu 4 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_1k_384.pth \
--test-size 384 CARS


### ConvNext (ConvNext-B + Inet22K + 384 * 384)
### CIFAR10
python train.py \
--save-dir CIFAR10/convnext_base_384_Inet22K_BS48_epoch50_lr002 \
--gpu 5 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_224.pth \
--test-size 384 CIFAR10

### CIFAR100
python train.py \
--save-dir CIFAR100/convnext_base_384_Inet22K_BS48_epoch50_lr002 \
--gpu 5 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_224.pth \
--test-size 384 CIFAR100

### CUB
python train.py \
--save-dir CUB/convnext_base_384_Ine22K_BS48_epoch150_lr002 \
--gpu 5 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_224.pth \
--test-size 384 CUB

### CARS
python train.py \
--save-dir CARS/convnext_base_384_Inet22K_BS48_epoch150_lr002 \
--gpu 5 \
--model convnext_base \
--max-lr 0.002 \
--warmup-epoch 1 \
--nb-epoch 150 \
--train-size 384 \
--batch-size 48 \
--pretrained-net ./pretrained_weight/convnext_base_22k_224.pth \
--test-size 384 CARS