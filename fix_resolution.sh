## CUB 224
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CUB/resnet50_inet_224_500e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CUB

## CUB 448
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CUB/resnet50_inet_448_500e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 448 \
--test-size 448 CUB

## CUB 224 --> 448
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CUB/resnet50_inet_224_400e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 400 \
--train-size 224 \
--test-size 224 CUB

python train.py \
--pretrained-net ./CUB/resnet50_inet_224_400e/best_acc_net_ema.pth \
--save-dir CUB/FT_resnet50_inet_448_100e/ \
--gpu 0 \
--model resnet50 \
--max-lr 0.001 \
--nb-epoch 100 \
--train-size 448 \
--test-size 448 CUB


## CARS 224
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CARS/resnet50_inet_224_500e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 224 \
--test-size 224 CARS

## CARS 448
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CARS/resnet50_inet_448_500e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 500 \
--train-size 448 \
--test-size 448 CARS

## CARS 224 --> 448
python train.py \
--pretrained-net ./pretrained_weight/resnet50_inet_torch_sota.pth \
--save-dir CARS/resnet50_inet_224_400e \
--gpu 0 \
--model resnet50 \
--max-lr 0.005 \
--nb-epoch 400 \
--train-size 224 \
--test-size 224 CARS

python train.py \
--pretrained-net ./CARS/resnet50_inet_224_400e/best_acc_net_ema.pth \
--save-dir CARS/FT_resnet50_inet_448_100e/ \
--gpu 0 \
--model resnet50 \
--max-lr 0.001 \
--nb-epoch 100 \
--train-size 448 \
--test-size 448 CARS