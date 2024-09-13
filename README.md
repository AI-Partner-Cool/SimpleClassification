# A simple tutorial to Image Classification


* 95.62% top-1 acc on CIFAR10, 79.11% top-1 acc on CIFAR10 and CIFAR100 with resnet18 

* Support CIFAR10, CIFAR100, CUB and CARS

* Support standard retrained weight on Inet1K in supervised and self-supervised fashion (MOCOV2) 

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

* CUB, CARS are trained with Inet1K Pretrained weight

* Report result with EMA

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th colspan="3">Stage 1</th>
      <th colspan="3">Stage 2</th>
      <th colspan="2">Test Size = 224 * 224</th>
      <th colspan="2">Test Size = 448 * 448</th>
    </tr>
    <tr>
      <th>LR</th>
      <th>Epoch</th>
      <th>Size</th>
      <th>LR</th>
      <th>Epoch</th>
      <th>Size</th>
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
      <td><strong>224 * 224</strong></td>
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
      <td><strong>448 * 448</strong></td>
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
      <td><strong>224 * 224</strong></td>
      <td>0.001</td>
      <td>100</td>
      <td><strong>448 * 448</strong></td>
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

## 关注我们


* [公众号文章：极简图像分类，95.58% on CIFAR10 with ResNet18](https://mp.weixin.qq.com/s/d557nluTn_PLfpsmYiodCQ)

* 关注我们的公众号：AI上分搭子

<p align="center" width="100%">
    <img width="25%" src="https://github.com/AI-Partner-Cool/SimpleClassification/blob/main/data/qrcode_wechat.bmp">
</p>