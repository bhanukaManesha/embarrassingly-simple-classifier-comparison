from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, RMSprop
import torch.nn.functional as F
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on : {device}')

    network = IndoorNetwork()
    network.to(device)
    print(network)

    exp_name = 'network-1e5'

    # resume_training = True
    # resume_exp_name = exp_name
    # resume_epoch = 300

    resume_training = True
    resume_exp_name = exp_name
    resume_epoch = 900


    indoorscene_traindataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TrainImages.txt',
        feature_file='Dataset/features.h5',
        train=True)
    train_loader = DataLoader(indoorscene_traindataset, batch_size=16, shuffle=True, num_workers=1)

    indoorscene_testdataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TestImages.txt',
        feature_file='Dataset/features.h5',
        train=False)
    val_loader = DataLoader(indoorscene_testdataset, batch_size=16, shuffle=True, num_workers=1)


    optimizer = RMSprop(network.parameters(), lr=1e-5)

    if resume_training:
        checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
        state = torch.load(checkpoint, map_location=torch.device('cpu'))
        network.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Resuming from checkpoint : {checkpoint}")

    for epoch in range(1000):
        total_loss = 0
        total_correct = 0
        total_preds = []
        total_labels = []

        total_val_loss = 0
        total_val_correct = 0
        total_val_preds = []
        total_val_labels = []

        for batch in train_loader: # Get Batch
            images = batch[0].to(device)
            labels = batch[1].to(device)

            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss

            optimizer.zero_grad()
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
            total_preds.append(preds.squeeze(dim=0))
            total_labels.append(labels.squeeze(dim=0))

        # validation
        for batch in val_loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            val_preds = network(images) # Pass Batch
            val_loss = F.cross_entropy(val_preds, labels)

            total_val_loss += val_loss.item()
            total_val_correct += get_num_correct(val_preds, labels)
            total_val_preds.append(val_preds.squeeze(dim=0))
            total_val_labels.append(labels.squeeze(dim=0))

        total_preds = torch.cat(total_preds).argmax(dim=1).to('cpu')
        total_labels = torch.cat(total_labels).to('cpu')

        total_val_preds = torch.cat(total_val_preds).argmax(dim=1).to('cpu')
        total_val_labels = torch.cat(total_val_labels).to('cpu')

        target_names = indoorscene_traindataset.mapping

        print("---------------------")
        print(
            "epoch:", epoch,
            "loss:", total_loss,
            "accuracy:", accuracy_score(total_labels, total_preds),
            "precision:", precision_score(total_labels, total_preds, average='weighted'),
            "recall:", recall_score(total_labels, total_preds, average='weighted'),
            "f1-score", f1_score(total_labels, total_preds, average='weighted')
        )
        if epoch % 100 == 0:
            print(classification_report(total_labels, total_preds, target_names=target_names))


        print(
            "epoch:", epoch,
            "val_loss:", total_val_loss,
            "val_accuracy:", accuracy_score(total_val_labels, total_val_preds),
            "val_precision:", precision_score(total_val_labels, total_val_preds, average='weighted'),
            "val_recall:", recall_score(total_val_labels, total_val_preds, average='weighted'),
            "val_f1-score", f1_score(total_val_labels, total_val_preds, average='weighted')
        )

        if epoch % 100 == 0:
            print(classification_report(total_val_labels, total_val_preds, target_names=target_names))

        if epoch % 100 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, f'checkpoints/{exp_name}-{epoch}'
            )
            print("Created checkpoint")


if __name__ == '__main__':
    main()


