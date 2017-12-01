# SSD: Single Shot MultiBox Detector

[![Build Status](https://travis-ci.org/weiliu89/caffe.svg?branch=ssd)](https://travis-ci.org/weiliu89/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](https://www.linkedin.com/in/dragomiranguelov), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com).

### 简介

SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325) and our [slide](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf).

<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">
</p>

| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes | Input resolution
|:-------|:-----:|:-------:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | ~6000 | ~1000 x 600 |
| [YOLO (customized)](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 | 448 x 448 |
| SSD300* (VGG16) | 77.2 | 46 | 8732 | 300 x 300 |
| SSD512* (VGG16) | **79.8** | 19 | 24564 | 512 x 512 |


<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px">
</p>

_Note: SSD300* and SSD512* are the latest models. Current code should reproduce these results._

### 引用 SSD

Please cite SSD in your publications if it helps your research:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }

### Contents
1. [安装](#installation)
2. [预备](#preparation)
3. [训练/评估](#traineval)
4. [模型](#models)

### 安装
1. 下载代码。假设把Caffe克隆到目录`$CAFFE_ROOT`下   
```终端   
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Build 代码. 按照 [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) 安装
  必要的packages，然后build。      
  ```终端   
  # 根据Caffe安装的方式修改Makefile.config。
  cp Makefile.config.example Makefile.config
  make -j8
  # 确保include $CAFFE_ROOT/python到PYTHONPATH环境变量内.
  make py
  make test -j8
  # 运行测试，可选   
  make runtest -j8
  ```

### 预备
1. 下载 [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). 假设文件被下载到了`$CAFFE_ROOT/models/VGGNet/`目录   

2. 下载VOC2007和VOC2012数据集. 假设下载到了`$HOME/data/`目录   
```Shell
  # 下载数据.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # 解压数据.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

3. 创建LMDB文件.   
```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # 如有必要，可以按需修改create_data.sh文件.
  # 编码trainval和test原始图像，生成lmdb文件:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### 训练/评估   
1. 训练你自己的模型并评估.   
```Shell
  # 创建模型定义文件并保存模型训练快照到如下路径:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
  # 保存当前评估结果到:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
  # 120K次迭代之后，应该可以达到77.*的mAP.
  python examples/ssd/ssd_pascal.py
  ```
  如果不乐意自己训练模型，可以在[here](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA)下载预训练好的模型.注意是用PASCAL VOC数据集训练的。

2. 使用最新模型快照评估模型.   
```Shell
  # 如果你需要对训练的模型进行评估，执行脚本:
  python examples/ssd/score_ssd_pascal.py
  ```

3. 使用webcam摄像头测试模型. 注意: 按 <kbd>esc</kbd> 停止.   
```Shell
  # 连接webcam摄像头和预训练的模型进行演示，运行:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) 展示了一个在 [MSCOCO](http://mscoco.org) 数据集上训练的模型SSD500的演示视频.

4. 查看 [`examples/ssd_detect.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) 或者 [`examples/ssd/ssd_detect.cpp`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_detect.cpp) 如何使用ssd模型检测物体. 查看 [`examples/ssd/plot_detections.py`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/plot_detections.py) 如何绘制 ssd_detect.cpp的检测结果.

5. 如果使用其他数据集训练, 请参考data/OTHERDATASET 了解更多细节. 目前支持COCO 和 ILSVRC2016数据集. 建议使用 [`examples/ssd.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) 检查新的数据集是否符合要求.

### 模型
在不同数据集上训练了模型以供下载. 为了复现论文[Table 6](https://arxiv.org/pdf/1512.02325v4.pdf)中的结果, 每个模型文件夹内都包含一个`.caffemodel` 文件, 几个`.prototxt` 文件, 以及python脚本文件.

1. PASCAL VOC 模型:
   * 07+12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_ZDIxVHBEcUNBb2s)
   * 07++12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WnR2T1BGVWlCZHM), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_MjFjNTlnempHNWs)
   * COCO<sup>[1]</sup>: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_NDlVeFJDc2tIU1k), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_TW4wTC14aDdCTDQ)
   * 07+12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_UFpoU01yLS1SaG8), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_X3ZXQUUtM0xNeEk)
   * 07++12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_TkFPTEQ1Z091SUE), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_NVVNdWdYNEh1WTA)

2. COCO 模型:
   * trainval35k: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_dlJpZHJzOXd3MTg)

3. ILSVRC 模型:
   * trainval1: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_a2NKQ2d1d043VXM), [SSD500](https://drive.google.com/open?id=0BzKzrI_SkD1_X2ZCLVgwLTgzaTQ)

<sup>[1]</sup>We use [`examples/convert_model.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/convert_model.ipynb) to extract a VOC model from a pretrained COCO model.
