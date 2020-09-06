
import torch.nn as nn
import torch.nn.functional as F


class IndoorNetwork(nn.Module):

    def __init__(self):
        super(IndoorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=8192)
        self.out = nn.Linear(in_features=8192, out_features=67)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.out(x), dim=1)



