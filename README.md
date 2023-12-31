# DetOH: 仅使用热图的无锚点目标检测器

## 介绍

DetOH是一个基于热图的无锚点目标检测器，其主要特点是只使用热图来进行目标检测，而不需要锚点或者候选框。该检测器不仅准确性高，而且计算速度快，适用于实时目标检测。
## 环境要求
- Python 3.8 及以上版本
- PyTorch 1.9.1 及以上版本
- CUDA 11.4 及以上版本（如果您想使用GPU进行加速）
- NumPy库
- COCOEval库（用于计算目标检测的coco mAP）
  
请确保您的计算机满足以上要求，并安装了所需的依赖项，以便成功运行DetOH目标检测器。
## 代码

- `README.md`：本文件，提供了项目的介绍和使用说明。
- `backbonds.py`：该文件包含了骨干网络ResNet的实现代码。
- `shuffle_v2.py`：该文件包含了骨干网络ShuffleNet/Shuffle-UNet的实现代码。
- `model.py`：该文件包含了多个特征提取模型的实现代码，包括DenseNet/Unet/FiboDense/DenseFibo等模型。
- `heatmap.py`：该文件包含了生成热图的代码。
- `mAP.py`：该文件包含了计算平均精度（mean Average Precision，mAP）的代码。
- `match.cpp`：该文件包含了匹配目标框和真实框的C++代码。
- `DetOH.py`：该文件包含了DetOH模型训练的实现代码。
- `png2hdf5.py`：该文件包含了将PNG格式的图像转换为HDF5格式的代码。
- `predict.py`：该文件包含了使用训练好的模型进行目标检测的代码。
- `temp.py`：该文件包含了一些临时性的代码，供开发和调试使用。
- `track.py`：该文件包含了使用惯性滤波进行目标跟踪的代码。

## 使用

要使用DetOH进行目标检测，您需要执行以下步骤：

1. 准备数据集并将其转换为所需格式。
2. 使用`heatmap.py`中的代码生成热图。
3. 使用`DetOH.py`中的代码训练模型。
4. 使用`predict.py`中的代码进行目标检测。
5. 使用`mAP.py`中的代码计算模型的平均精度。
   
请注意，训练DetOH模型可能需要较长时间，具体时间取决于您使用的数据集和计算机性能。

## 引用

如果您在研究中使用了DetOH，请引用以下论文：Wu Ruohao, et al. DetOH: An Anchor-Free Object Detector with Only Heatmaps[C]//International Conference on Advanced Data Mining and Applications. Cham: Springer Nature Switzerland, Shenyang, China, August 21-23, 2023: 152-167.
