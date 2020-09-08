from nn import IndoorResNetNetwork, IndoorMnasnetNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adamax, Adam, SGD, lr_scheduler
import torch.nn.functional as F
import torch
from metrics import Metrics
from pathlib import Path
from utils import results
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on : {device}')

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def evaluate(network, data_loader, device):
    # Evaluate
    pred = []
    y_true = []

    for batch in data_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        y_preds = network(images)
        pred.append(y_preds.squeeze(dim=0))
        y_true.append(labels.squeeze(dim=0))

    y_pred = torch.cat(pred)
    y_true = torch.cat(y_true)

    return y_true, y_pred

def run(params):

    if params['feature_extractor'] == 'resnext101':
        network = IndoorResNetNetwork()
    else:
        network = IndoorMnasnetNetwork()

    network.to(device)

    resume_training = False
    resume_exp_name = params['exp_name']
    resume_epoch = 900

    # Create checkpoints folder
    path = f'results/{params["model_type"]}/{params["exp_name"]}/'
    Path(path).mkdir(parents=True, exist_ok=True)


    indoorscene_traindataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TrainImages.txt',
        feature_file=f'Dataset/{params["feature_extractor"]}-features.h5',
        train=True)
    train_loader = DataLoader(indoorscene_traindataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)

    indoorscene_testdataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TestImages.txt',
        feature_file=f'Dataset/{params["feature_extractor"]}-features.h5',
        train=False)
    val_loader = DataLoader(indoorscene_testdataset, batch_size=params['batch_size'], shuffle=True, num_workers=1)

    optimizer = None
    if params['optimizer'] == 'adamax':
        optimizer = Adamax(network.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adam':
        optimizer = Adam(network.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'sgd':
        optimizer = SGD(network.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

    lr_scheduler = None
    if params['learning_rate_decay']:
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=params['learning_rate_decay_rate'])

    classes = indoorscene_traindataset.mapping

    train_metrics, val_metrics = Metrics(classes), Metrics(classes)

    if resume_training:
        checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
        network.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Resuming from checkpoint : {checkpoint}")

    best_train_loss = 9999
    best_val_loss = 9999
    best_train_correct = 0
    best_val_correct = 0


    best_train_model_name = ""
    best_train_path = ""
    best_val_model_name = ""
    best_val_path = ""

    for epoch in range(params['epochs']):
        total_loss = 0
        total_correct = 0
        total_preds = []
        total_labels = []

        total_val_loss = 0
        total_val_correct = 0
        total_val_preds = []
        total_val_labels = []

        desc = f'{epoch}/{params["epochs"] - 1} | train: [{best_train_correct:.3f} :: {best_train_loss:.3f}] | val_loss: [{best_val_correct:.3f} :: {best_val_loss:.3f}] '

        for batch in tqdm(train_loader, desc=desc):
            images = batch[0].to(device)
            labels = batch[1].to(device)

            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss

            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_loss += loss.item()
            total_preds.append(preds.squeeze(dim=0))
            total_labels.append(labels.squeeze(dim=0))

        # validation
        for batch in val_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            val_preds = network(images) # Pass Batch
            val_loss = F.cross_entropy(val_preds, labels)

            total_val_loss += val_loss.item()
            total_val_preds.append(val_preds.squeeze(dim=0))
            total_val_labels.append(labels.squeeze(dim=0))

        total_preds = torch.cat(total_preds).argmax(dim=1).to('cpu')
        total_labels = torch.cat(total_labels).to('cpu')
        total_correct += accuracy_score(total_labels, total_preds)

        total_val_preds = torch.cat(total_val_preds).argmax(dim=1).to('cpu')
        total_val_labels = torch.cat(total_val_labels).to('cpu')
        total_val_correct += accuracy_score(total_val_labels, total_val_preds)

        # update metrics
        train_metrics.update_epoch(epoch, total_loss, total_labels, total_preds)
        val_metrics.update_epoch(epoch, total_val_loss, total_val_labels, total_val_preds)

        # Step the learning rate
        if params['learning_rate_decay'] and epoch % 10 == 9:
            lr_scheduler.step()


        # Save the best train model
        if total_loss < best_train_loss:

            if os.path.exists(best_train_model_name):
                os.remove(best_train_model_name)

            best_train_loss = total_loss
            best_train_correct = total_correct

            best_train_path = f'results/{params["model_type"]}/{params["exp_name"]}/train-best/'
            Path(best_train_path).mkdir(parents=True, exist_ok=True)

            best_train_model_name = f'{best_train_path}/train-best-{epoch}'

            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_train_model_name
            )

            # x_true, x_pred = evaluate(network, train_loader, device)
            # y_true, y_pred = evaluate(network, val_loader, device)
            #
            # name = f'{params["model_type"]}-{params["exp_name"]}-train-best-{epoch}'
            # results(x_true, x_pred, y_true, y_pred, classes, params, path=best_train_path, name=name)
            # train_metrics.save(params, best_train_path, 'train-log')
            # val_metrics.save(params, best_train_path, 'val-log')


        # Save the best val model
        if total_val_loss < best_val_loss:
            if os.path.exists(best_val_model_name):
                os.remove(best_val_model_name)

            best_val_loss = total_val_loss
            best_val_correct = total_val_correct

            best_val_path = f'results/{params["model_type"]}/{params["exp_name"]}/val-best/'
            Path(best_val_path).mkdir(parents=True, exist_ok=True)

            best_val_model_name = f'{best_val_path}/val-best-{epoch}'

            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_val_model_name
            )

            # x_true, x_pred = evaluate(network, train_loader, device)
            # y_true, y_pred = evaluate(network, val_loader, device)
            #
            # name = f'{params["model_type"]}-{params["exp_name"]}-val-best-{epoch}'
            # results(x_true, x_pred, y_true, y_pred, classes, params, path=best_val_path, name=name)
            # train_metrics.save(params, best_val_path, 'train-log')
            # val_metrics.save(params, best_val_path, 'val-log')

    # Save the latest model
    latest_path = f'results/{params["model_type"]}/{params["exp_name"]}/latest-path/'
    Path(latest_path).mkdir(parents=True, exist_ok=True)
    torch.save({
            'epoch': params['epochs']-1,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'{latest_path}latest-{params["epochs"]-1}'
    )

    x_true, x_pred = evaluate(network, train_loader, device)
    y_true, y_pred = evaluate(network, val_loader, device)

    name = f'{params["model_type"]}-{params["exp_name"]}-latest-{params["epochs"]-1}'
    results(x_true, x_pred, y_true, y_pred, classes, params, path=latest_path, name=name)
    train_metrics.save(params, latest_path, 'train-log')
    val_metrics.save(params, latest_path, 'val-log')


    # Save the best training model
    state = torch.load(best_train_model_name, map_location=torch.device(device))
    network.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch = state['epoch']

    x_true, x_pred = evaluate(network, train_loader, device)
    y_true, y_pred = evaluate(network, val_loader, device)

    name = f'{params["model_type"]}-{params["exp_name"]}-train-best-{epoch}'
    results(x_true, x_pred, y_true, y_pred, classes, params, path=best_train_path, name=name)
    train_metrics.save(params, best_train_path, 'train-log')
    val_metrics.save(params, best_train_path, 'val-log')


    # Save the best val model
    state = torch.load(best_val_model_name, map_location=torch.device(device))
    network.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    epoch = state['epoch']

    x_true, x_pred = evaluate(network, train_loader, device)
    y_true, y_pred = evaluate(network, val_loader, device)

    name = f'{params["model_type"]}-{params["exp_name"]}-val-best-{epoch}'
    results(x_true, x_pred, y_true, y_pred, classes, params, path=best_val_path, name=name)
    train_metrics.save(params, best_val_path, 'train-log')
    val_metrics.save(params, best_val_path, 'val-log')








def run_loop():

    type = 'nn'
    # feature_extractors = ['resnext101', 'mnasnet1_0']
    # batch_sizes = [2,4,8,16,32]
    # learning_rates = [1e-1, 1e-3, 1e-5]
    # optimizers = ['adamax', 'adam', 'sgd']
    # learning_rate_decays = [True, False]

    feature_extractors = ['mnasnet1_0', 'resnext101', ]
    batch_sizes = [32]
    learning_rates = [1e-5]
    optimizers = ['adamax', 'adam', 'sgd']
    learning_rate_decays = [True, False]

    for feature_extractor in feature_extractors:
        for batch_size in batch_sizes:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for learning_rate_decay in learning_rate_decays:
                        expt_name = f'{feature_extractor}-{batch_size}-{optimizer}-{learning_rate}{"-wd" if learning_rate_decay else ""}'
                        print(type, expt_name)

                        params = {
                            "batch_size": batch_size,
                            "epochs": 2,
                            "learning_rate": learning_rate,
                            "weight_decay": 1e-5,
                            "learning_rate_decay_rate":0.96,
                            "learning_rate_decay" : learning_rate_decay,
                            "optimizer": optimizer,
                            'feature_extractor': feature_extractor,
                            'exp_name': expt_name,
                            'model_type': type
                        }

                        run(params)




if __name__ == '__main__':


    run_loop()


