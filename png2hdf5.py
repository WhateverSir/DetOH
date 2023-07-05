import os
import io
import h5py
import numpy as np
from PIL import Image
base_path = './images' # 图片存放文件夹
save_path = './h5_dataset.hdf5'
### 建立一个字典，保存 name: label的映射
class_to_idx = {} #
classes = sorted(entry.name for entry in os.scandir(base_path) if entry.is_dir())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
print('dict generating done')
H5File = h5py.File(save_path, 'a')
for identity_name in os.listdir(base_path):
    identity_dir =  os.path.join(base_path, identity_name)
    group_identity = H5File.create_group(identity_name)
    # 找到指定文件夹下的所有图片
    for img_name in os.listdir(identity_dir):
        img_path = os.path.join(identity_dir, img_name)
        ### 预处理，这里可以自己定义
        img = Image.open(img_path).convert('L')     ### 预处理，转化为灰度图片
        img = img.resize((128,128), Image.BILINEAR) ### 预处理，Resize 
        img = np.array(img)
        sample = group_identity.create_group('.'.join(img_name.split('.')[:-1]))
        sample.create_dataset('img', data = img)
        sample.create_dataset('label', data = class_to_idx[identity_name])
H5File.close()
###加载HDF5数据集
class MyDataset(data.Dataset):
    def __init__(self, archive,image='image',mask='mask'):
        self.archive = h5py.File(archive, 'r')
        self.data = self.archive[image]
        self.labels = self.archive[mask]

    def __getitem__(self, index):
        image = self.data[index]
        mask = self.get_multi_class_labels(self.labels[index])
        return image, mask

    def __len__(self):
        return len(self.labels)

    def get_multi_class_labels(self,truth, n_labels=3, labels=(0, 1, 2)):
        new_shape =  [n_labels, ]+list(truth.shape[1:])
        y = np.zeros(new_shape, np.int8)
        for label_index in range(n_labels):
            if labels is not None:
                y[label_index, :, :][truth[0, :, :] == labels[label_index]] = 1
            else:
                y[label_index, :, :][truth[0, :, :] == label_index] = 1
        return y

    def close(self):
        self.archive.close()

### 创建 DataLoader
train_data = MyDataset('xxxx.h5',image='image',mask='mask')
train_loader = DataLoader(dataset=train_data,
                                  num_workers=0,
                                  batch_size=8,
                                  shuffle=True)
### 输出创建的数据集
diter = iter(train_loader)
for i, data in enumerate(train_loader):
    print(i, data[0].shape, data[1])
print('Datasets done')