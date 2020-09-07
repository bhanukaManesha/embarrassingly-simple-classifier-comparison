
import torch.nn as nn
import torch.nn.functional as F


class IndoorNetwork(nn.Module):

    def __init__(self):
        super(IndoorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.out = nn.Linear(in_features=512, out_features=67)

    def forward(self, x):
        x = self.dropout1(F.relu(self.batchnorm1(self.fc1(x))))
        x = F.relu(self.batchnorm2(self.fc2(x)))
        return F.log_softmax(self.out(x), dim=1)



