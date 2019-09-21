import numpy as np
import torch.utils.data as data
import os
from imageio import imread
from PIL import Image
from torchvision import transforms


class MyDataset(data.Dataset):
    def __init__(self, data_path='../gesture', train=True, transform=None):
        self.data_path = data_path
        self.train = train
        self.names, self.labels = self.__dataset_info()

    def __dataset_info(self):
        if self.train:
            data_path = os.path.join(self.data_path, 'train')
        else:
            data_path = os.path.join(self.data_path, 'test')
        annotation_file = os.path.join(data_path, 'annotations.txt')
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            names = []
            labels = []
            for line in lines:
                names.append(line.split()[0])
                labels.append(np.array(line.split()[1]))
        return np.array(names), np.array(labels).astype(np.long)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if self.train:
            data_path = os.path.join(self.data_path, 'train')
        else:
            data_path = os.path.join(self.data_path, 'test')

        img = imread(os.path.join(data_path, self.names[index]))
        img = Image.fromarray(img)
        img = img.convert('L')
        img = img.resize((320, 120))
        tran = transforms.ToTensor()
        img = tran(img)
        label = self.labels[index]
        return img, label





