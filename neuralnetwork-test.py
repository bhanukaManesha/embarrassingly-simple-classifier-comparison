from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from utils import results
import torch
from torch.optim import RMSprop
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
    y_true = []
    for batch in val_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        val_preds = network(images)
        final_pred.append(val_preds.squeeze(dim=0))
        y_true.append(labels.squeeze(dim=0))
    y_pred = torch.cat(final_pred)
    y_true = torch.cat(y_true)

    classes = indoorscene_traindataset.mapping

    results(y_true, y_pred, classes)

if __name__ == '__main__':
    main()