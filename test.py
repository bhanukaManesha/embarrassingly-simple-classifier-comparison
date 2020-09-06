from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, plot_multiclass_roc
import torch
from torch.optim import RMSprop
import numpy as np
from sklearn.metrics import classification_report
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on : {device}')

    network = IndoorNetwork()
    network.to(device)
    print(network)
    optimizer = RMSprop(network.parameters(), lr=1e-5)

    # Params
    resume_exp_name = 'network-1e5'
    resume_epoch = 900

    # Load the model
    checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
    state = torch.load(checkpoint, map_location=torch.device('cpu'))
    network.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    print(f"Resuming from checkpoint : {checkpoint}")

    # Evaluate
    final_pred = []
    final_labels = []
    for batch in val_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        val_preds = network(images)
        final_pred.append(val_preds.squeeze(dim=0))
        final_labels.append(labels.squeeze(dim=0))
    y_pred = torch.cat(final_pred)
    final_pred = y_pred.argmax(dim=1)
    final_labels = torch.cat(final_labels)

    print(final_labels.shape)
    print(final_pred.shape)

    classes = indoorscene_traindataset.mapping

    #classification report
    print(classification_report(final_labels, final_pred, target_names=classes))

    # AUC curve
    y_true = np.zeros((len(indoorscene_testdataset), 67))
    for idx, lbl in enumerate(final_labels):
        y_true[idx][lbl] = 1

    y_pred = y_pred.detach().numpy()
    plot_multiclass_roc(y_true,y_pred, classes=classes)

    # Confusion matrix
    cm = confusion_matrix(final_labels, final_pred)
    plot_confusion_matrix(cm, classes)

if __name__ == '__main__':
    main()