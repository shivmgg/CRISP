# CRISP

[![built with Python3.6](https://img.shields.io/badge/build%20with-python%203.6-red.svg)](https://www.python.org/)
[![built with PyTorch1.4](https://img.shields.io/badge/build%20with-pytorch%201.4-brightgreen.svg)](https://pytorch.org/)

### Introduction

In this repository you will find a pytorch implementation of CRISP for three models. 

### Getting Started

When using anaconda virtual environment all you need to do is run the following 
command and conda will install everything for you. 
See [environment.yml](./environment.yml):

    conda env create --file environment.yml
    conda activate crisp-env
    
To reproduce the results on the ResNet-50 benchmark you just
 need to run the following code:

```
chmod +x run_resnet_imagenet.sh
./run_resnet_imagenet.sh
```

Feel free to change the model and dataset type in the script. 
