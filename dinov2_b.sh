# ### CARS + DINOV2-B

python train.py \
--save-dir CARS/dinov2_base_patch14_reg_378_BS32_epoch50_Lr2e5 \
--gpu 9 \
--model dinov2_base_patch14_reg \
--max-lr 2e-5 \
--min-lr 1e-7 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 378 \
--batch-size 32 \
--optimizer adamw \
--test-size 378 CARS


### CUB + DINOV2-B
python train.py \
--save-dir CUB/dinov2_base_patch14_reg_378_BS32_epoch50_Lr2e5 \
--gpu 9 \
--model dinov2_base_patch14_reg \
--max-lr 2e-5 \
--min-lr 1e-7 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 378 \
--batch-size 32 \
--optimizer adamw \
--test-size 378 CUB

## CIFAR10 + DINOV2-B
python train.py \
--save-dir CIFAR10/dinov2_base_patch14_reg_378_BS32_epoch50_Lr2e5 \
--gpu 9 \
--model dinov2_base_patch14_reg \
--max-lr 2e-5 \
--min-lr 1e-7 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 378 \
--batch-size 32 \
--optimizer adamw \
--test-size 378 CIFAR10

## CIFAR100 + DINOV2-B
python train.py \
--save-dir CIFAR100/dinov2_base_patch14_reg_378_BS32_epoch50_Lr2e5 \
--gpu 9 \
--model dinov2_base_patch14_reg \
--max-lr 2e-5 \
--min-lr 1e-7 \
--warmup-epoch 1 \
--nb-epoch 50 \
--train-size 378 \
--batch-size 32 \
--optimizer adamw \
--test-size 378 CIFAR100
