import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch
import os
import h5py
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class IndoorSceneDataset(Dataset):
    def __init__(self, text_file, root_dir, transform=None):
        super(IndoorSceneDataset).__init__()
        self.indoor_scenes = pd.read_csv(text_file, header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.indoor_scenes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.indoor_scenes.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        indoor_scene = self.indoor_scenes.iloc[idx, 0].split('/')[0]
        if self.transform:
            image = self.transform(image)


        return image, indoor_scene

class IndoorSceneFeatureDataset(Dataset):
    def __init__(self, text_file, feature_file, root_dir, transform=None):
        super(IndoorSceneDataset).__init__()

        self.indoor_scenes = pd.read_csv(text_file, header=None)

        f = h5py.File(feature_file, 'r')
        self.features = f['features']

        self.root_dir = root_dir
        self.transform = transform

        self.label_encoder = LabelEncoder()
        self.labels = self.__get_labels()


    def __len__(self):
        return len(self.indoor_scenes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.features[idx].squeeze(0)
        indoor_scene = self.labels[idx]
        return image, indoor_scene

    def __get_labels(self):

        scenes = []
        for key, item in tqdm(self.indoor_scenes.iterrows()):
            scene = item[0].split('/')[0]
            scenes.append(scene)

        unq_labels = list(set(scenes))
        print(len(unq_labels))
        self.label_encoder.fit(unq_labels)

        return self.label_encoder.transform(scenes)



if __name__ == '__main__':
    # indoorscene_dataset = IndoorSceneDataset(text_file='Dataset/TrainImages.txt',
    #                                     root_dir='Dataset/Images/',
    #                                     transform=transforms.Compose([
    #                                            transforms.Resize((224,224)),
    #                                             transforms.ToTensor(),
    #                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #                                        ]))
    #
    # trainloader = DataLoader(indoorscene_dataset, batch_size=1, shuffle=True, num_workers=4)
    #
    # for i_batch, (images, labels) in enumerate(trainloader):
    #     print(images.shape)
    #     print(labels)
    #
    #     # observe 4th batch and stop.
    #     if i_batch == 4:
    #         break

    indoorscene_dataset = IndoorSceneFeatureDataset(
                                            text_file='Dataset/TrainImages.txt',
                                            feature_file = 'Dataset/features.h5',
                                             root_dir='Dataset/Images/',
                                             transform=transforms.Compose([
                                                 transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                             ]))

    trainloader = DataLoader(indoorscene_dataset, batch_size=8, shuffle=True, num_workers=1)

    for i_batch, (images, labels) in enumerate(trainloader):
        print(images.shape)
        print(labels)

        # observe 4th batch and stop.
        if i_batch == 4:
            break
