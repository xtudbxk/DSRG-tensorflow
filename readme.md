### Introduction

This is a project which just move the [DSRG](https://github.com/speedinghzl/DSRG) to tensorflow. The DSRG is referring to the approach for weakly-supervised semantic segmentation in the paper ["Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing"](https://github.com/speedinghzl/DSRG). And here, I just use the tensorflow to implement the approach with the help of the [DSRG](https://github.com/kolesman/SEC) project.

### Citing this repository

If you find this code useful in your research, please consider citing them:

> @inproceedings{kolesnikov2016seed,  
>
> ​    title={Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing},
>
> ​    author={Huang, Zilong and Wang, Xinggang and Wang, Jiasi and Liu, Wenyu and Wang, Jingdong},
>
> ​    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
>
> ​    pages={7014--7023},
>
> ​    year={2018}
>
> }

### Preparation

for using this code, you have to do something else:

##### 1. Install pydensecrf

For using the densecrf in python, we turn to the project [pydensecrf](https://github.com/lucasb-eyer/pydensecrf). And you just using the following code to install it.

> pip install pydensecrf

##### 2. Download the data and model

1. for pascal data, please referring to its [official website](http://host.robots.ox.ac.uk/pascal/VOC/). Just download it and extract it in the data/ .
2. for localization_cues-cal.pickle, please referring to [DSRG](https://github.com/speedinghzl/DSRG). And download it and extract it in the data/.
3. for init.model, you can download it from [googledriver](https://drive.google.com/file/d/1kxDguwRaIDm5WS6JTNzi8GO-HqKJqKnm/view) or [BaiduNetdisk](https://pan.baidu.com/s/1Q1wmAX7Do9jvvLMt3_8tFw). Just download it and put it in model/.

For more details, you can referring to the correspond code files or leave a message in the issue.

### Training

then, you just input the following sentence to train it.

> python DSRG.py <gpu_id>

### Result

train the network with totally 24 epoch and lr=1e-3 for the begining, then lr drops into one tenth of its old value each 8 epoches. And in testing, we first resize the featmap into 321x321 and update the crf config for testing( just remove the 12 in the origin crf config), And the final result is 0.564 in the validatation set while it is 0.574(without pretrain step) in the paper.
