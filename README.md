# CIFAR10 + ResNet18 95.51% Acc






## Table of Content
* [Dependencies](#Dependencies)

* [Baseline](#Baseline)


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

## 95.58 Acc on ResNet18

Using the following setup, ResNet18 achieved 95.58% accuracy. The experiment completed in 1 hour on a single RTX 3090 GPU.

```bash
python train.py --model resnet18 --max-lr 5e-2 --save-dir maxLr5e2_resnet18
```

* Training log : `maxLr5e2_resnet18/run.log`

* Tensorboard monitoring is [here](maxLr5e2_resnet18/events.out.tfevents.1724169146.localhost.localdomain.78424.0)


