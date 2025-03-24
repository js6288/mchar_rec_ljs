from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 这两行的代码是为了实现定长字符识别，因为数据集每张图像中字符的长度并不一致，
        # 这里是将字符长度统一为5个字符，少于5个字符的补0（类别10代表数字0）
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl)  + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)