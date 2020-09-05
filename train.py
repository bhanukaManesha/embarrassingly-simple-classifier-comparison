from nn import IndoorNetwork
from dataset import IndoorSceneFeatureDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on : {device}')

    network = IndoorNetwork()
    network.to(device)
    print(network)

    exp_name = 'network-1e5'
    train = False
    resume_training = True
    resume_exp_name = exp_name
    resume_epoch = 6


    indoorscene_traindataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TrainImages.txt',
        feature_file='Dataset/train-features.h5')
    train_loader = DataLoader(indoorscene_traindataset, batch_size=16, shuffle=True, num_workers=1)

    indoorscene_testdataset = IndoorSceneFeatureDataset(
        text_file='Dataset/TestImages.txt',
        feature_file='Dataset/test-features.h5')
    val_loader = DataLoader(indoorscene_testdataset, batch_size=16, shuffle=True, num_workers=1)


    optimizer = Adam(network.parameters(), lr=1e-5)

    if resume_training:
        checkpoint = f'checkpoints/{resume_exp_name}-{resume_epoch}'
        state = torch.load(checkpoint)
        network.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f"Resuming from checkpoint : {checkpoint}")

    if train:
        for epoch in range(1000):
            total_loss = 0
            total_correct = 0

            total_val_loss = 0
            total_val_correct = 0

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

            # validation
            for batch in val_loader:
                images = batch[0].to(device)
                labels = batch[1].to(device)

                val_preds = network(images) # Pass Batch
                val_loss = F.cross_entropy(val_preds, labels)

                total_val_loss += val_loss.item()
                total_val_correct += get_num_correct(val_preds, labels)

            print(
                "epoch:", epoch,
                "loss:", total_loss,
                "accuracy:", total_correct/len(indoorscene_traindataset),
                "val_loss:", total_val_loss,
                "val_accuracy:", total_val_correct / len(indoorscene_testdataset)
            )

            if epoch % 2 == 0:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, f'checkpoints/{exp_name}-{epoch}'
                )
                print("Created checkpoint")


    # Evaluate
    final_pred = []
    final_labels = []
    for batch in val_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)

        val_preds = network(images)
        final_pred.append(val_preds.squeeze(dim=0))
        final_labels.append(labels.squeeze(dim=0))

    final_pred = torch.cat(final_pred).argmax(dim=1)
    final_labels = torch.cat(final_labels)

    cm = confusion_matrix(final_labels, final_pred)

    classes = indoorscene_traindataset.mapping
    plot_confusion_matrix(cm, classes)




if __name__ == '__main__':
    main()


