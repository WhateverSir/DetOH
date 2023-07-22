# DetOH: 仅使用热图的无锚点目标检测器

## 介绍

DetOH是一个基于热图的无锚点目标检测器，其主要特点是只使用热图来进行目标检测，而不需要锚点或者候选框。该检测器不仅准确性高，而且计算速度快，适用于实时目标检测。

## 代码

- `README.md`：本文件，提供了项目的介绍和使用说明。
- `backbonds.py`：该文件包含了DetOH中使用的一种新型卷积操作的实现代码。
- `heatmap.py`：该文件包含了生成热图的代码。
- `mAP.py`：该文件包含了计算平均精度（mean Average Precision，mAP）的代码。
- `match.cpp`：该文件包含了匹配目标框和真实框的C++代码。
- `mycenternet.py`：该文件包含了DetOH模型的实现代码。
- `png2hdf5.py`：该文件包含了将PNG格式的图像转换为HDF5格式的代码。
- `predict.py`：该文件包含了使用训练好的模型进行目标检测的代码。
- `temp.py`：该文件包含了一些临时性的代码，供开发和调试使用。
- `track.py`：该文件包含了使用卡尔曼滤波进行目标跟踪的代码。

## 使用

要使用DetOH进行目标检测，您需要执行以下步骤：

1. 准备数据集并将其转换为HDF5格式。
2. 使用`mycenternet.py`中的代码训练模型。
3. 使用`predict.py`中的代码进行目标检测。
4. 使用`mAP.py`中的代码计算模型的平均精度。

## 参考文献

- Zhou, X., Wang, D., & Krähenbühl, P. (2020). Objects as points. arXiv preprint arXiv:1904.07850.
- Zhou, X., Wang, D., & Krähenbühl, P. (2019). Bottom-up object detection by grouping extreme and center points. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 850-859.
