import json
import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from code.jessie_utils import data_dir
import random
import torch
class DigitsDataset(Dataset):
    """

    DigitsDataset

    Params:
      data_dir(string): data directory

      label_path(string): label path

      aug(bool): wheather do image augmentation, default: True  是否要做图片增强
    """

    def __init__(self, mode='train', size=(128, 256), aug=True):
        super(DigitsDataset,self).__init__()
        # 是否要做图片增强
        self.aug = aug

        self.size = size
        # 当前模式：训练、验证、测试
        self.mode = mode
        self.width = 224

        self.batch_count = 0
        if mode == 'test': #测试集没有label
            self.imgs = glob(data_dir['test_data']+'*.png')
            self.labels = None

        else:
            labels = json.load(open(data_dir['%s_label' % mode], 'r'))

            imgs = glob(data_dir['%s_data' % mode]+'*.png')

            # imgs是全部img的路径，os.path.split(img)[-1]取路径的最后一位，也就是img名字，对应json的key. 把每个图片路径和对应的json结合起来
            self.imgs = [(img, labels[os.path.split(img)[-1]]) for img in imgs \
                         if os.path.split(img)[-1] in labels]


    def __getitem__(self, idx):
        # 如果是训练和验证模式，则取图片和label
        if self.mode != 'test':
            img, label = self.imgs[idx]
        else: #测试模式
            img = self.imgs[idx]
            label = None
        # 打开图片
        img = Image.open(img)
        # 不管要不要做图片增强，transforms都要做的事情
        trans0 = [
            transforms.ToTensor(), # 转化为tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
        # img.size[0]：图片的宽度，img.size[1]图片的高度，这段原作者的代码不知为何没有用到
        min_size = self.size[0] if (img.size[1] / self.size[0]) < (img.size[0] / self.size[1]) else self.size[1]

        trans1 = [
            transforms.Resize(128),
            transforms.CenterCrop((128,self.width)),# 中心裁剪,参数表示裁剪后的尺寸
        ]
        # 是否要做图片增强
        if self.aug:
            trans1.extend([
                # 改变亮度、对比度、饱和度
                transforms.ColorJitter(0.1, 0.1, 0.1),
                # 以一定的概率将图片转换成灰度图
                transforms.RandomGrayscale(0.1),
                # 对图像进行随机的仿射变换
                transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5)
            ])
        # 添加公共操作
        trans1.extend(trans0)

        if self.mode != 'test':
            return transforms.Compose(trans1)(img), torch.tensor(
                # 为了实现定长字符识别，因为数据集每张图像中字符的长度并不一致，
                # 这里是将字符长度统一为5个字符，少于5个字符的补0（类别10代表数字0）
                label['label'][:4] + (4 - len(label['label'])) * [10]).long()
        else:
            return transforms.Compose(trans1)(img), self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

    # 动态调整训练过程中图像的裁剪宽度 ，并通过控制 batch_count 实现周期性的参数更新。
    def collect_fn(self, batch):
        imgs, labels = zip(*batch)
        if self.mode == 'train': #在训练模式中
            # 每处理 10 个批次 从 range(224, 256, 16) 中随机选择一个新的宽度值（如 224, 240, 256 等），赋值给 self.width
            if self.batch_count > 0 and self.batch_count %10 ==0:
                self.width = random.choice(range(224,256,16))
        # 记录批次计数
        self.batch_count += 1
        return torch.stack(imgs).float(),torch.stack(labels)
