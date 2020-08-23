import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import torch
import os
import plotly.graph_objs as go

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



if __name__ == '__main__':
    indoorscene_dataset = IndoorSceneDataset(text_file='Dataset/TrainImages.txt',
                                        root_dir='Dataset/Images/',
                                        transform=transforms.Compose([
                                               transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                           ]))

    trainloader = DataLoader(indoorscene_dataset, batch_size=1, shuffle=True, num_workers=4)

    for i_batch, (images, labels) in enumerate(trainloader):
        print(images.shape)
        print(labels)

        # observe 4th batch and stop.
        if i_batch == 0:
            break
