### CIFAR10 + ResNet18

python train.py \
--save-dir CIFAR10/resnet18 \
--gpu 0 \
--model resnet18_cifar \
--max-lr 0.05 \
--nb-epoch 200 \
--train-size 32 \
--test-size 32 CIFAR10

### CIFAR100 + ResNet18
python train.py \
--save-dir CIFAR100/resnet18 \
--gpu 0 \
--model resnet18_cifar \
--max-lr 0.05 \
--nb-epoch 200 \
--train-size 32 \
--test-size 32 CIFAR100

### CUB + ResNet18

python train.py \
--save-dir CUB/resnet18 \
--gpu 0 \
--model resnet18 \
--max-lr 0.05 \
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