# Changelog

This changelog's purpose is primarily to keep track on network additions and additionally on notable changes to the code base.

## 2021-12-09

### Mechanics
- add AIO/TF profilers support
- add --num_runs arg to benchmark runners


## 2021-11-19

### Cosmetics
- rebrand DLS to AIO


## 2021-11-03

### Networks
- 3D Unet [BraTS] in fp32 and fp16

### Miscellaneous
- update requirements.txt
- test scripts updated to have pre-defined num of runs


## 2021-10-14

### Networks
- 3D Unet in fp32

### Miscellaneous
- PYTHONPATH to model_zoo no longer needed to be specified by user


## 2021-10-06

### Networks
- Inception ResNet v2, Inception v2, NasNet-Mobile, VGG-16, VGG-19 in FP32, FP16 and INT8
- Inception v4, ResNet-101 v2 in FP32 and FP16
- ResNet-50 v2, MobileNet v1, Inception v3, Squeezenet in FP32 and INT8
- NasNet-Large in INT8


## 2021-10-01

### Networks
- SSD ResNet-34 in FP16


## 2021-09-16

### Networks
- BERT Large (from mlcommons:inference repo) in FP32, FP16

### Added
- Squad v1.1 dataset support


## 2021-09-09

### Networks
- SSD ResNet-34 in FP32


## 2021-05-06

### Networks
- SSD Inception v2 in FP32, FP16
- YOLO v4 Tiny in FP32

### Added
- Changelog
- Download of tarballs


## <2021-05-06

### Networks
- SSD MobileNet v2 in FP32, INT8
- MobileNet v2 in FP32, FP16, INT8
- DenseNet-169 in FP32, FP16, INT8
- ResNet-50 v1.5 in FP32, FP16, INT8

### Added
- Whole mechanics for running things in benchmark and test modes
