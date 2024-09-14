# Image Classification Tutorial


* 95.62% top-1 acc on CIFAR10, 79.11% top-1 acc on CIFAR10 and CIFAR100 with resnet18 

* Support CIFAR10, CIFAR100, CUB and CARS

* Support standard retrained weight on Inet1K in supervised and self-supervised fashion ([MOCOV2](https://arxiv.org/abs/2003.04297)) 

* [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)

* Tensorboard Visualization, EMA, [ONLY simple data aug](https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/dataloader.py#L13-L18): flip, random crop, normalization


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

## Data 

| Dataset  | Nb CLS | No. Training | No. Test |
|----------|--------|--------------|----------|
| CIFAR10  | 10     | 50,000       | 10,000   |
| CIFAR100 | 100    | 50,000       | 10,000   |
| CUB      | 200    | 5,994        | 5,794    |
| CARS     | 196    | 8,144        | 8,041    |

One can directly go to `./data/`, launch `download.sh` to download CUB and CARS dataset: 

* Downloading CUB from [cyizhuo's repo](https://github.com/cyizhuo/CUB-200-2011-dataset)

* Downloading CARS from [cyizhuo's repo](https://github.com/cyizhuo/Stanford-Cars-dataset)

## Baseline Results on CUB, CARS, CIFAR10 and CIFAR100

* CUB, CARS are trained and tested with **224 * 224**
* CIFAR10, CIFAR100 are trained and tested with **32 * 32**

| LR   | EMA  | Arch      | CUB   | CARS  | CIFAR10 | CIFAR100 |
|------|------|-----------|-------|-------|---------|----------|
| 0.05 | -    | ResNet18  | 60.74 | 87.33 | 95.31   | **79.13**    |
| 0.05 | TRUE | ResNet18  | **64.64** | **87.54** | **95.62**   | 79.11    |
| 0.05 | -    | ResNet50  | 57.16 | 88.87 | -       | -        |
| 0.05 | TRUE | ResNet50  | **63.43** | **89.27** | -       | -        |

* EMA improves in most cases

Reproducing the above exp with: 

```bash
  bash baseline.sh
  ```

## Pretraining matters

* CUB, CARS are trained and tested with **224 * 224**

* Report result with EMA
 
| LR    | Pretrained | Arch      | CUB   | CARS  |
|-------|------------|-----------|-------|-------|
| 0.05  | -          | ResNet18  | 64.64 | 87.54 |
| 0.005 | Inet1K       | ResNet18  | **77.11** | **88.42** |
|-------|------------|-----------|-------|-------|
| 0.05  | -          | ResNet50  | 63.43 | 89.27 |
| 0.005 | Inet1K       | ResNet50  | **84.47** | 91.38 |
| 0.005 | MocoV2     | ResNet50  | 79.01 | **92.33** |

* Pretraining significantly improves the accuracy

* MocoV2 (SSL pretrained) might be promising for some small datasets, e.g. CARS.

Reproducing the above exp with: 

```bash
  cd pretrained_weight/
  bash download.sh
  cd ..
  bash pretraining.sh
  ```

## [Fixing the train-test resolution discrepancy](https://arxiv.org/abs/1906.06423)

* One can refer to [[经典论文] Meta的FixRes (NeurIPS 2019)](https://mp.weixin.qq.com/s?__biz=MzkwODczNTIyNw==&mid=2247483869&idx=1&sn=e35be8947ca05650fc25a409bd3a50b2&chksm=c0c42649f7b3af5f6283ec3905548901451128294bd9361cd0181a62c4a8c4bc154bef50264e#rd)

* CUB, CARS are trained with Inet1K Pretrained weight

* Report result with EMA

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th colspan="3">Stage 1</th>
      <th colspan="3">Stage 2</th>
      <th colspan="2">Test Size 224*224</th>
      <th colspan="2">Test Size 448*448</th>
    </tr>
    <tr>
      <th>LR</th>
      <th>Epoch</th>
      <th>Train Size</th>
      <th>LR</th>
      <th>Epoch</th>
      <th>Train Size</th>
      <th>CUB</th>
      <th>CARS</th>
      <th>CUB</th>
      <th>CARS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0.005</td>
      <td>500</td>
      <td><strong>224*224</strong></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td><strong>84.47</strong></td>
      <td><strong>91.38</strong></td>
      <td>73.86</td>
      <td>91.58</td>
    </tr>
    <tr>
      <td>0.005</td>
      <td>500</td>
      <td><strong>448*448</strong></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>78.56</td>
      <td>79.23</td>
      <td>85.71</td>
      <td>91.88</td>
    </tr>
    <tr>
      <td>0.005</td>
      <td>400</td>
      <td><strong>224*224</strong></td>
      <td>0.001</td>
      <td>100</td>
      <td><strong>448*448</strong></td>
      <td>82.59</td>
      <td>87.46</td>
      <td><strong>86.07</strong></td>
      <td><strong>92.18</strong></td>
    </tr>
  </tbody>
</table>

* Training and testing with the same resolution achieves more stable accuracy

* Training on a small resolution then finetuning on larger resolution manage to achieve consistent gain

* Check [paperswithcode](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200), these results **86.07** on CUB and **92.18** on CARS are not bad.

Reproducing the above exp with: 

```bash
  cd pretrained_weight/
  bash download.sh
  cd ..
  bash fix_resolution.sh
  ```

## 关注我们，其他感兴趣的内容


* [公众号文章：极简图像分类，95.58% on CIFAR10 with ResNet18](https://mp.weixin.qq.com/s/d557nluTn_PLfpsmYiodCQ)

* [[经典论文] Meta的FixRes (NeurIPS 2019)](https://mp.weixin.qq.com/s?__biz=MzkwODczNTIyNw==&mid=2247483869&idx=1&sn=e35be8947ca05650fc25a409bd3a50b2&chksm=c0c42649f7b3af5f6283ec3905548901451128294bd9361cd0181a62c4a8c4bc154bef50264e#rd)

* [[经典论文] 让卷积再次具有平移不变性 (2019 ICML)](https://mp.weixin.qq.com/s?__biz=MzkwODczNTIyNw==&mid=2247483808&idx=1&sn=6b823ed172d5617f6ad48a6700d2aa65&chksm=c0c42634f7b3af22f4ea57a94b670993817c9808300e234bb5db4785693558e50f712df55343#rd)

* [[Pytorch实战] model.eval()究竟干了啥](https://mp.weixin.qq.com/s?__biz=MzkwODczNTIyNw==&mid=2247483827&idx=1&sn=0f86c601ea7c5268789f0c466a3bf7ed&chksm=c0c42627f7b3af3173d8743a2d859b15b458b87dd58592c94e89937c37e41edd8bea5de796b0#rd)

* [[Pytorch实战] 什么是In-place操作](https://mp.weixin.qq.com/s?__biz=MzkwODczNTIyNw==&mid=2247483835&idx=1&sn=15c4fd19a6e457b55abe77868cb52393&chksm=c0c4262ff7b3af3990f5e5aade4ad3c47be1004d99628074c6376756c00076178b296b2b1add#rd)


* 关注我们的公众号：AI上分搭子

<p align="center" width="100%">
    <img width="25%" src="https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/data/qrcode_wechat.bmp">
</p>
