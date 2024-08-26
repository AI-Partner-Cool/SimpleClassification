# ResNet18 95.58% Acc with ResNet18 on Cifar10


* 1 hour on a single GPU

* Cosine learning rate scheduer

* Tensorboard 

* [simple data aug](https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/dataloader.py#L13-L18): flip, random crop, normalization


## Dependencies

The model can be trained on a single GPU with more than 12 GB of memory.

- Install PyTorch adapted to your CUDA version via Conda:
  ```bash
  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
  ```
 
- Install TensorBoard: 
  ```bash
  conda install matplotlib tensorboard
  ```

## 95.58% Accuracy on ResNet18

Using the following setup, ResNet18 achieved 95.58% accuracy. The experiment completed in 1 hour on a single RTX 3090 GPU.

```bash
python train.py --model resnet18 --max-lr 5e-2 --save-dir maxLr5e2_resnet18
```

* Training log : `maxLr5e2_resnet18/run.log`

* Tensorboard monitoring is [here](maxLr5e2_resnet18/events.out.tfevents.1724169146.localhost.localdomain.78424.0)


<p align="center" width="100%">
    <img width="50%" src="https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/data/tensorboard.png">
</p>

## ResNet18 v.s. PreactResNet18 

| ResNet18    | PreactResNet18 |
| -------- | ------- |
| 95.58  | 95.44   |

## 关注我们


* [公众号文章：极简图像分类，95.58% on CIFAR10 with ResNet18](https://mp.weixin.qq.com/s/d557nluTn_PLfpsmYiodCQ)

* 关注我们的公众号：AI上分搭子

<p align="center" width="100%">
    <img width="25%" src="https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/data/qrcode_wechat.bmp">
</p>