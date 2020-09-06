from __future__ import print_function, division

import torch
import numpy as np
from torchvision import datasets, models, transforms
from dataset import IndoorSceneDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
from sklearn.preprocessing import LabelEncoder

class FeatureExtractor():

    def __init__(self, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        # Load the pretrained resnet model
        self.model = models.resnext50_32x4d(pretrained=True)
        print(self.model)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # print(self.model)
        # Set model to evaluation mode
        self.model.eval()

        self.hdf5 = None

        self.label_encoder = LabelEncoder()
        self.mapping = None

        self.train_features = None
        self.train_labels = None

        self.test_features = None
        self.test_labels = None


        self.__extract_features(dataset='train')
        self.__extract_features(dataset='test')


    def __extract_features(self, dataset):

        embeddings = []
        labels = []

        if dataset == 'train':
            dataloader = self.train_dataloader
        else:
            dataloader = self.test_dataloader

        for i_batch, (images, l) in enumerate(tqdm(dataloader)):

            embedding = torch.zeros(dataloader.batch_size, 2048)

            def copy_data(self, input, output):
                reshaped_output = output.data.view(-1,2048)
                embedding.copy_(reshaped_output)

            h = self.layer.register_forward_hook(copy_data)

            self.model(images)

            h.remove()

            numpy_embedding = embedding.detach().numpy()

            labels.append(l[0])

            embeddings.append(numpy_embedding)

        # Create a feature vector
        if dataset == 'train':
            self.train_features = np.asarray(embeddings)
        else:
            self.test_features = np.asarray(embeddings)

        if self.mapping is None:
            # Convert to indexes
            self.mapping = list(set(labels))
            self.label_encoder.fit(self.mapping)

        if dataset == 'train':
            self.train_labels = self.label_encoder.transform(labels)
        else:
            self.test_labels = self.label_encoder.transform(labels)

    def to_hdf5(self, path):
        self.hdf5 = h5py.File(path, 'w')
        self.hdf5.create_dataset('train_features', data=self.train_features)
        self.hdf5.create_dataset('train_labels', data=self.train_labels)

        self.hdf5.create_dataset('test_features', data=self.test_features)
        self.hdf5.create_dataset('test_labels', data=self.test_labels)

        mapping_list = [n.encode("utf-8") for n in self.mapping]
        self.hdf5.create_dataset('mapping', (len(mapping_list), 1), 'S30', mapping_list)

        self.hdf5.close()

if __name__ == '__main__':
    train_indoorscene_dataset = IndoorSceneDataset(text_file='Dataset/TrainImages.txt',
                                                  root_dir='Dataset/Images/',
                                                  transform=transforms.Compose([
                                                      transforms.Resize((224, 224)),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225]),
                                                  ]))


    test_indoorscene_dataset = IndoorSceneDataset(text_file='Dataset/TestImages.txt',
                                        root_dir='Dataset/Images/',
                                        transform=transforms.Compose([
                                               transforms.Resize((224,224)),
                                                transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                           ]))

    trainloader = DataLoader(train_indoorscene_dataset, batch_size=1, shuffle=False, num_workers=0)
    testloader = DataLoader(test_indoorscene_dataset, batch_size=1, shuffle=False, num_workers=0)


    fe = FeatureExtractor(train_dataloader=trainloader, test_dataloader=testloader)
    fe.to_hdf5('Dataset/features.h5')