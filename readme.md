### Introduction

This is a project which just move the [DSRG](https://github.com/speedinghzl/DSRG) to tensorflow. The DSRG is referring to the approach for weakly-supervised semantic segmentation in the paper ["Weakly-Supervised Semantic Segmentation Network with Deep Seeded Region Growing"](https://github.com/speedinghzl/DSRG). And here, I just use the tensorflow to implement the approach with the help of the [DSRG](https://github.com/speedinghzl/DSRG) project.

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

1. for pascal data, please referring to its [official website](http://host.robots.ox.ac.uk/pascal/VOC/)  and to the augmental [SBD data](http://home.bharathh.info/pubs/codes/SBD/download.html). Just download it and extract it in the data/, then 'cd data' and run convert.py with 'python convert.py'.
2. for localization_cues-cal.pickle, please referring to [DSRG](https://github.com/speedinghzl/DSRG) or [BaiduNetdisk](https://pan.baidu.com/s/14a94qw4nBulqqKHLMbQGUg)(which fetching code is qgig). And download it and extract it in the data/.
3. for init.model, you can download it from [googledriver](https://drive.google.com/file/d/1kxDguwRaIDm5WS6JTNzi8GO-HqKJqKnm/view) or [BaiduNetdisk](https://pan.baidu.com/s/1Q1wmAX7Do9jvvLMt3_8tFw). Just download it and put it in model/.

For more details, you can referring to the correspond code files or leave a message in the issue.

### Training

then, you just input the following sentence to train it.

> python DSRG.py <gpu_id>

### Result
If you scale the input image with factors 0.5, 0.75 and 1.0, then use the max to merge the result. 
The final result is 0.567 in the validatation set while it is 0.574(without pretrain step) in the paper.

### Trained model
[google_driver](https://drive.google.com/open?id=1hlSl1EaDKWA00hvd6Ar0xDZ9uOZ7HKYu)
[BaiduNetdisk](https://pan.baidu.com/s/1vITyeBR5kxaGcOF0BHGkJA)

### Evaluation
I just release a [project](https://github.com/xtudbxk/semantic-segmentation-metrics) to provide the code for evaluation.
