import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)

        self.dropout = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        self.output_layer = nn.Linear(800, 10)


    def forward(self, x):

        tensor = self.conv(x)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout(tensor)

        tensor = self.conv2(tensor) 
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout(tensor)

        tensor = torch.flatten(tensor, 1)
        tensor = self.output_layer(tensor)
        output = F.log_softmax(tensor, dim = 1)
        return output


class PaperCNN(nn.Module):

    def __init__(self, numClass):
        super(PaperCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)

        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.25)

        self.fc1 = nn.Linear(1*1*256, 64)
        self.fc2 = nn.Linear(1*1*64, 10)


    def forward(self, x):

        tensor = self.conv1(x)
        tensor = F.relu(tensor)

        tensor = self.conv2(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout1(tensor)

        tensor = self.conv3(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout2(tensor)

        tensor = self.conv4(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, 2)
        tensor = self.dropout3(tensor)

        tensor = torch.flatten(tensor, 1)

        tensor = self.fc1(tensor)
        tensor = F.relu(tensor)

        tensor = self.fc2(tensor)
        output = F.log_softmax(tensor, dim = 1)
        return output

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=11, padding=2, stride=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x