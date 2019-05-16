# Video-Object-Detection (Still in progress)
## Description
Based on the paper: "Towards High Performance Video Object Detection" use Pytorch 0.4.1 and Python 3.6

The model is currently running on Bosch Traffic Light Dataset only, but it will be easy to add another dataset by modifying dataloader.

For training simply use 'python main.py' and set args according to your need.

## Reference Links
The RefineDet's code is inspired by https://github.com/lzx1413/PytorchSSD.

The vgg pretrained model is downloaded from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth.

The FlowNet's code and pretrained model is inspired and downloaded from https://github.com/NVIDIA/flownet2-pytorch.

## Others
Some self-understanding about the paper's design (may be some are wrong, I will go deeper into those points later):
* In training process, two frames are randomly selected with former one as key-frame and later one as non-key-frame.
* The "q_propagate" factor is normalized to 0-1 during training, in case for preserving gradient and allow backward propagating. In inference process, it should be either 0 or 1.
* is_aggr and is_partial flag is both set to True for all frame-pairs during training since each batch has few key-frame. In inference process they should be treated differently.

Some self-modification:
* I use RefineDet instead of single ResNet as base detection network. Therefore, the results of flownet is also used in some middle source layers in addition to final layer.

TODO:
* add multiple gpu's support. (Forgive I am a beginner to Pytorch orz.)
* optimize the distribution of tensors that on CPU or GPU.
* add inference part.
* ...
