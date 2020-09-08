from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adamax
import torch.nn.functional as F
import torch
from metrics import Metrics

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on : {device}')

    network = IndoorNetwork()
    network.to(device)
    print(network)
    print(params)

    resume_training = False
    resume_exp_name = params['exp_name']
    resume_epoch = 900

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

    optimizer = Adamax(network.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    classes = indoorscene_traindataset.mapping

    train_metrics, val_metrics = Metrics(classes), Metrics(classes)

    if resume_training:
        checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
        network.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Resuming from checkpoint : {checkpoint}")

    for epoch in range(params['epochs']):
        total_loss = 0
        total_preds = []
        total_labels = []

        total_val_loss = 0
        total_val_preds = []
        total_val_labels = []

        best_train_loss = 9999
        best_val_loss = 999

        for batch in train_loader: # Get Batch
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

        total_val_preds = torch.cat(total_val_preds).argmax(dim=1).to('cpu')
        total_val_labels = torch.cat(total_val_labels).to('cpu')

        # update metrics
        train_metrics.update_epoch(epoch, total_loss, total_labels, total_preds)
        val_metrics.update_epoch(epoch, total_val_loss, total_val_labels, total_val_preds)

        # Save the best train model
        if total_loss < best_train_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/{params["exp_name"]}/train-best-{epoch}'
            )

            x_true, x_pred = evaluate(network, train_loader, device)
            y_true, y_pred = evaluate(network, val_loader, device)

            train_metrics.save(x_true, x_pred, y_true, y_pred, params)

        # Save the best val model
        if total_val_loss < best_val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/{params["exp_name"]}/val-best-{epoch}'
            )

            x_true, x_pred = evaluate(network, train_loader, device)
            y_true, y_pred = evaluate(network, val_loader, device)

            train_metrics.save(x_true, x_pred, y_true, y_pred, params)


    # Save the latest model
    torch.save({
            'epoch': params['epochs']-1,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/{params["exp_name"]}/latest-{params["epochs"]-1}'
    )

    x_true, x_pred = evaluate(network, train_loader, device)
    y_true, y_pred = evaluate(network, val_loader, device)

    train_metrics.save(x_true, x_pred, y_true, y_pred, params)

def run_loop():

    type = 'nn'
    feature_extractors = ['resnext101', 'mnasnet1_0']
    batch_size = [2,4,8,16,32]
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    optimizer = ['adamax', 'adam', 'sgd']
    epochs = 500


    for feature_extractor in feature_extractors:
        for kernel in kernels:
            for c in C:
                expt_name = f'{feature_extractor}-{kernel}-{c}'
                print(type, expt_name)

                run({
                    'C': c,
                    'kernel': kernel,
                    'feature_extractor': feature_extractor,
                    'exp_name': expt_name,
                    'model_type': type
                })

    params = {
        "batch_size" : 4,
        "epochs" : 1000,
        "learning_rate":2e-5,
        "weight_decay":1e-5

    }



if __name__ == '__main__':


    run_loop()


