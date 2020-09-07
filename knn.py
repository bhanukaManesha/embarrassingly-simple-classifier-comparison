from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
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
    classes = indoorscene_traindataset.mapping

    # training
    clf = KNeighborsClassifier(
        n_neighbors=len(classes),
        algorithm=params['algorithm'],
        weights=params['weight'],
        n_jobs=4
    )
    train_start_time = time.time()
    clf = clf.fit(train_images, train_labels)
    train_time = time.time() - train_start_time

    classes = indoorscene_traindataset.mapping

    # train metrics
    train_pred_start_time = time.time()
    x_pred = clf.predict(train_images)
    train_pred_time = time.time() - train_pred_start_time

    x_pred_ohe = torch.zeros((len(x_pred), len(classes)))
    for idx, lbl in enumerate(x_pred):
        x_pred_ohe[idx][lbl] = 1

    # test metrics
    test_images, y_true = next(iter(val_loader))
    test_pred_start_time = time.time()
    y_pred = clf.predict(test_images)
    test_pred_time = time.time() - test_pred_start_time
    y_pred_ohe = torch.zeros((len(y_pred), len(classes)))
    for idx, lbl in enumerate(y_pred):
        y_pred_ohe[idx][lbl] = 1

    # Saving metrics
    params['train_time'] = train_time
    params['train_pred_time'] = train_pred_time
    params['test_pred_time'] = test_pred_time

    # Store the results
    results(train_labels, x_pred_ohe, y_true, y_pred_ohe, classes, params)


def run_loop():
    type = 'knn'
    feature_extractors = ['resnext101', 'mnasnet1_0']
    algorithms = ['ball_tree','kd_tree','brute']
    weights = ['uniform', 'distance']

    for feature_extractor in feature_extractors:
        for algorithm in algorithms:
            for weight in weights:
                expt_name = f'{feature_extractor}-{algorithm}-{weight}'
                print(type, expt_name)

                run({
                    'weight':weight,
                    'algorithm':algorithm,
                    'feature_extractor':feature_extractor,
                    'exp_name' : expt_name,
                    'model_type':type
                })



if __name__ == '__main__':
    run_loop()