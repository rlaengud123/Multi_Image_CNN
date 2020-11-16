import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from itertools import chain
from sklearn.model_selection import train_test_split
import pickle
import torch
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, pathDirData, pathDirCSV, dataset, transform):

        self.root_dir = pathDirData
        self.transform = transform
        self.ImagePaths = []

        df = pd.read_csv(pathDirCSV)
        df.drop([0, 1, 37], axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['path'] = df['Name[tree]']
        df['OASPL_0yaw[float]'] = df['OASPL_0yaw[float]'].astype(float)

        for i in range(len(df)):
            df['path'][i] = 'run{}.png'.format(i + 1)


        train, self.test = train_test_split(df, test_size=0.3)
        self.train, self.val = train_test_split(train, test_size=0.3)

        if dataset == 'train':
            self.ImagePaths = list(self.train.path)
            self.Label = list(self.train.iloc[:,11])

        elif dataset == 'val':
            self.ImagePaths = list(self.val.path)
            self.Label = list(self.val.iloc[:,11])

        elif dataset == 'test':
            self.ImagePaths = list(self.test.path)
            self.Label = list(self.test.iloc[:,11])



    def __getitem__(self, index):

        imagePath = self.ImagePaths[index]
        Label = torch.FloatTensor(np.array(self.Label[index]))

        img_name = os.path.join(self.root_dir, 'APLR_iso_view', imagePath)
        image1 = Image.open(img_name).convert('RGB')

        img_name = os.path.join(self.root_dir, 'front_view', imagePath)
        image2 = Image.open(img_name).convert('RGB')

        img_name = os.path.join(self.root_dir, 'mirror_front_view', imagePath)
        image3 = Image.open(img_name).convert('RGB')

        img_name = os.path.join(self.root_dir, 'side_view', imagePath)
        image4 = Image.open(img_name).convert('RGB')

        img_name = os.path.join(self.root_dir, 'top_view', imagePath)
        image5 = Image.open(img_name).convert('RGB')

        sample = {'image1': image1, 'image2' : image2, 'image3' : image3, 'image4' : image4,
                  'image5' : image5, 'y': Label}

        sample['image1'] = self.transform(sample['image1'])
        sample['image2'] = self.transform(sample['image2'])
        sample['image3'] = self.transform(sample['image3'])
        sample['image4'] = self.transform(sample['image4'])
        sample['image5'] = self.transform(sample['image5'])

        return sample


    def __len__(self):
        return len(self.ImagePaths)
