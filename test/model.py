import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class CNNLSTM(nn.Module):

    def __init__(self, num_classes):
        super(CNNLSTM, self).__init__()
        self.hidden_size = 9
        self.conv1 = nn.Conv2d(1, 1, 3, stride=2)
        self.pool = nn.MaxPool2d(2, 1)
        # self.conv2 = nn.Conv2d(6, 16, 3)
        # conv dimensions/filters multiplied by output of pool height and width
        self.fc1 = nn.Linear(1 * 2 * 2, 9)
        self.lstm = nn.LSTM(input_size=9, hidden_size=self.hidden_size,
                            num_layers=1, batch_first=True)
        self.fc_1 = nn.Linear(self.hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        # num layers, batch size, hidden size
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        # x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size)) # hidden state
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_size))  # internal state
        x = x[None, :]
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out)  # first Dense
        return out


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
