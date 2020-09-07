from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.svm import SVC
from utils import results
import torch
import time

def run(params):
    indoorscene_traindataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TrainImages.txt',
        feature_file=f'Dataset/{params["feature_extractor"]}-features.h5',
        train=True)
    train_loader = DataLoader(indoorscene_traindataset, batch_size=len(indoorscene_traindataset), shuffle=True,
                              num_workers=1)

    indoorscene_testdataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TestImages.txt',
        feature_file=f'Dataset/{params["feature_extractor"]}-features.h5',
        train=False)
    val_loader = DataLoader(indoorscene_testdataset, batch_size=len(indoorscene_testdataset), shuffle=True,
                            num_workers=1)


    # training images
    train_images, train_labels = next(iter(train_loader))

    # training
    clf = SVC(
        C=params['C'],
        kernel=params['kernel'])
    train_start_time = time.time()
    clf = clf.fit(train_images, train_labels)
    train_time = time.time() - train_start_time

    classes = indoorscene_traindataset.mapping

    # train metrics
    pred_start_time = time.time()
    x_pred = clf.predict(train_images)
    pred_time = time.time() - pred_start_time
    x_pred_ohe = torch.zeros((len(x_pred), len(classes)))
    for idx, lbl in enumerate(x_pred):
        x_pred_ohe[idx][lbl] = 1
    results(train_labels, x_pred_ohe, classes, params['model_type'], params['expt_name'])

    # test metrics
    test_images, y_true = next(iter(val_loader))
    pred_start_time = time.time()
    y_pred = clf.predict(test_images)
    pred_time = time.time() - pred_start_time
    y_pred_ohe = torch.zeros((len(y_pred), len(classes)))
    for idx, lbl in enumerate(y_pred):
        y_pred_ohe[idx][lbl] = 1

    # Saving metrics
    params['train_time'] = train_time
    params['pred_time'] = pred_time

    results(y_true, y_pred_ohe, classes, params)





def run_loop():
    type = 'svm'
    feature_extractors = ['resnext101', 'mnasnet1_0']
    kernels = ['linear','poly','rbf', 'sigmoid','precomputed']
    C = [1.0, 2.0, 4.0, 8.0, 16.0]

    for feature_extractor in feature_extractors:
        for kernel in kernels:
            for c in C:
                expt_name = f'{feature_extractor}-{kernel}-{c}'
                print(type, expt_name)

                run({
                    'C':c,
                    'kernel':kernel,
                    'feature_extractor':feature_extractor,
                    'exp_name' : expt_name,
                    'model_type':type
                })


if __name__ == '__main__':
    run_loop()