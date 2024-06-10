import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
# from torchvision import models
# from torchinfo import summary

class YmshCNN(nn.Module):

    def __init__(self):
        super(YmshCNN, self).__init__()

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

class VideoCNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10):
        super(VideoCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.fc1 = nn.Linear(in_feature=16*5*5, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

class NumberModel(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(NumberModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(256 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x)) # [64, 1, 28, 28] -> [64, 32, 26, 26]
        x = nn.functional.relu(self.conv2(x)) # [64, 32, 26, 26] -> [64, 64, 26, 26]
        x = self.maxpool(x)                   # [64, 64, 26, 26] -> [64, 64, 13, 13]
        x = nn.functional.relu(self.conv3(x)) # [64, 64, 13, 13] -> [64, 128, 13, 13]
        x = self.maxpool(x)                   # [64, 128, 13, 13] -> [64, 128, 6, 6]
        x = self.dropout(x)
        x = nn.functional.relu(self.conv4(x)) # [64, 128, 6, 6] -> [64, 256, 6, 6]
        x = self.maxpool(x)                   # [64, 256, 6, 6] -> [64, 256, 3, 3]
        x = self.dropout(x)
        x = x.view(-1, 256 * 3 * 3)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class CharacterModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=52):
        super(CharacterModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjusted fully connected layer input size to match the 5x5 feature maps
        self.fc1 = nn.Linear(256 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [64, 1, 64, 64] -> [64, 32, 62, 62] -> [64, 32, 31, 31]
        x = F.dropout(x, 0.2)
        x = self.pool(F.relu(self.conv2(x)))  # [64, 32, 31, 31] -> [64, 64, 29, 29] -> [64, 64, 14, 14]
        x = F.dropout(x, 0.2)
        x = self.pool(F.relu(self.conv3(x)))  # [64, 64, 14, 14] -> [64, 128, 14, 14] -> [64, 128, 7, 7]
        x = F.dropout(x, 0.5)
        x = self.pool(F.relu(self.conv4(x)))  # [64, 128, 7, 7] -> [64, 256, 7, 7] -> [64, 256, 3, 3]
        x = F.dropout(x, 0.5)
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        return x

class SymbolModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=31):
        super(SymbolModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(256 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x)) # [64, 1, 45, 45] -> [64, 32, 43, 43]
        x = nn.functional.relu(self.conv2(x)) # [64, 32, 43, 43] -> [64, 64, 43, 43]
        x = self.maxpool(x)                   # [64, 64, 43, 43] -> [64, 64, 21, 21]
        x = nn.functional.relu(self.conv3(x)) # [64, 64, 21, 21] -> [64, 128, 21, 21]
        x = self.maxpool(x)                   # [64, 128, 21, 21] -> [64, 128, 10, 10]
        x = self.dropout(x)
        x = nn.functional.relu(self.conv4(x)) # [64, 128, 10, 10] -> [64, 256, 10, 10]
        x = self.maxpool(x)                   # [64, 256, 10, 10] -> [64, 256, 5, 5]
        x = self.dropout(x)
        x = x.view(-1, 256 * 5 * 5)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class LittleFishModel(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(LittleFishModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=num_classes)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv1(x)  # [64, 1, 28, 28] -> [64, 8, 26, 26]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # [64, 8, 26, 26] -> [64, 8, 13, 13]
        
        x = self.conv2(x)  # [64, 8, 13, 13] -> [64, 16, 11, 11]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # [64, 16, 11, 11] -> [64, 16, 5, 5]

        x = x.reshape(x.shape[0], -1)  # [64, 16, 5, 5] -> [64, 400]
        x = self.fc1(x)  # [64, 400] -> [64, 10]
        return x

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
    
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet5V1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14
            
            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5
            
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )
        
    def forward(self, x):
        return self.classifier(self.feature(x))
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Define the layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.25)
        
        self.fc2 = nn.Linear(512, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout4(x)
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


# model = NumberModel(in_channels=1, num_classes=10)
# summary(model, input_size=(1, 1, 28, 28), col_names=["input_size", "output_size", "num_params", "trainable"], depth=4)

# model = CharacterModel(in_channels=1, num_classes=10)
# summary(model, input_size=(1, 1, 64, 64), col_names=["input_size", "output_size", "num_params", "trainable"], depth=4)

# model = SymbolModel(in_channels=1, num_classes=10)
# summary(model, input_size=(1, 1, 45, 45), col_names=["input_size", "output_size", "num_params", "trainable"], depth=4)

