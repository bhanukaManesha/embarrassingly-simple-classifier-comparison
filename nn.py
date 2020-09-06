
import torch.nn as nn
import torch.nn.functional as F


class IndoorNetwork(nn.Module):

    def __init__(self):
        super(IndoorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=67)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return F.softmax(self.out(x), dim=1)



