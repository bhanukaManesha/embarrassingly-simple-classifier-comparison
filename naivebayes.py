from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.naive_bayes import GaussianNB
from utils import results
import torch

indoorscene_traindataset = IndoorSceneFeatureDataset(
    text_file='Dataset/TrainImages.txt',
    feature_file='Dataset/features.h5',
    train=True)
train_loader = DataLoader(indoorscene_traindataset, batch_size=len(indoorscene_traindataset), shuffle=True, num_workers=1)

indoorscene_testdataset = IndoorSceneFeatureDataset(
    text_file='Dataset/TestImages.txt',
    feature_file='Dataset/features.h5',
    train=False)
val_loader = DataLoader(indoorscene_testdataset, batch_size=len(indoorscene_testdataset), shuffle=True, num_workers=1)


def main():

    train_images, train_labels = next(iter(train_loader))

    clf = GaussianNB()
    clf = clf.fit(train_images, train_labels)

    test_images, y_true = next(iter(val_loader))

    y_pred = clf.predict(test_images)

    classes = indoorscene_traindataset.mapping

    y_pred_ohe = torch.zeros((len(y_pred), len(classes)))
    for idx, lbl in enumerate(y_pred):
        y_pred_ohe[idx][lbl] = 1

    results(y_true, y_pred_ohe, classes)



if __name__ == '__main__':
    main()