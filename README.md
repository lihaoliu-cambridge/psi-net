# Colorization
Make gray scale image colorful.

****
|Author|LIU Lihao|
|---|---
|E-mail|lhliu@cse.cuhk.edu.hk
****


## Introduction

Trained a U-net model for colorization tasks, as shown in the following figures.
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/2e089029ed09604fb02d66635466cce5.jpg)
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/5fda23ff7eba3203f7e7887570a6e755.jpg)
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/73456f7cb8ddc1d32d762d895bd3174a.jpg)
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/e67159e9efc1962cd2e334a022fe47d7.jpg)
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/f1b4ec389dae180cd5a00ae5d7717d62.jpg)
  ![image](https://github.com/CaptainWilliam/Colorization/blob/master/data/output_data/result/unet_1/test/240/f706932f62d986c1d27a3c14925e078e.jpg)



## Installation

pytorch: http://pytorch.org/

tensorboardX: https://github.com/lanpa/tensorboard-pytorch

Download and unzip this project: Colorization-master.zip.

Download dataset(places) into data folder and unzip it directly:
<br>https://drive.google.com/open?id=1PeP-UXtw85Vc75Lp8GgHly-A8hdqht1t


## Todos

 - Modify the [args.yaml](https://github.com/CaptainWilliam/Colorization/blob/master/conf/args.yaml), add the parameters your deep learning model need under the "running_params" item. Details are shown in another project: https://github.com/CaptainWilliam/Deep-Learning-Model-Saving-Helper
 - Pass the running_params (a python dict which contains the running parameters) to you own model.
 - The first parameter "is_training" is True for training mode, "is_training" is False for test mode.
 - Finish you mode(training or test), and run it.
 
```sh
$ cd Colorization-master
$ python main.py
```
