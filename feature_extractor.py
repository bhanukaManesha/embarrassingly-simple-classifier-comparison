from __future__ import print_function, division

import torch
import numpy as np
from torchvision import datasets, models, transforms
from dataset import IndoorSceneDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py

class FeatureExtractor():

    def __init__(self, dataloader):
        self.dataloader = dataloader

        # Load the pretrained resnet model
        # self.model = models.resnet18(pretrained=True)
        self.model = models.resnext50_32x4d(pretrained=True)
        print(self.model)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # print(self.model)
        # Set model to evaluation mode
        self.model.eval()

        self.hdf5 = None
        self.features = None

    def extract_features(self):

        embeddings = []

        for i_batch, (images, _) in enumerate(tqdm(self.dataloader)):

            embedding = torch.zeros(self.dataloader.batch_size, 2048)

            def copy_data(self, input, output):
                reshaped_output = output.data.view(-1,2048)
                embedding.copy_(reshaped_output)

            h = self.layer.register_forward_hook(copy_data)

            self.model(images)

            h.remove()

            numpy_embedding = embedding.detach().numpy()

            embeddings.append(numpy_embedding)

        # Create a feature vector
        self.features = np.asarray(embeddings)

    def to_hdf5(self, path):
        self.hdf5 = h5py.File(path, 'w')
        self.hdf5.create_dataset('features', data=self.features)
        self.hdf5.close()



if __name__ == '__main__':

    indoorscene_dataset = IndoorSceneDataset(text_file='Dataset/TrainImages.txt',
                                        root_dir='Dataset/Images/',
                                        transform=transforms.Compose([
                                               transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                           ]))

    trainloader = DataLoader(indoorscene_dataset, batch_size=1, shuffle=False, num_workers=0)


    fe = FeatureExtractor(dataloader=trainloader)
    fe.extract_features()
    fe.to_hdf5('Dataset/features.h5')