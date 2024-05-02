import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# (Blue, Green , Red)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


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


def Train(model, train_loder, epochs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, epochs+1):
        for batchIdx, (data, target) in enumerate(train_loder):
            optimizer.zero_grad()
            predict = model(data)
            loss = F.nll_loss(predict, target)
            loss.backward()
            optimizer.step()

            if batchIdx % 10 == 0:
                step = batchIdx + 1
                print(f"Train Epoch {epoch}: [{step * len(data)} / {len(train_loder.dataset)} ({int(100 * batchIdx / len(train_loder))}%)] loss: {round(loss.item(), 6)}")

    print("training completed")

def Test(model, test_loder):
    model.eval()
    correctCnt = 0
    with torch.no_grad():
        for data , target in test_loder:
            predict = model(data)
            answer = predict.argmax(dim = 1, keepdim = True)
            correctCnt += answer.eq(target.view_as(answer)).sum().item()
    print(f"Test Results: Accuracy: {int(100 * correctCnt / len(test_loder.dataset))}%")

batchSize = 64
epochs = 2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

trainDataSet = datasets.MNIST("./data",
                            train = True,
                            download = True,
                            transform = transform
)

testDataSet = datasets.MNIST("./data",
                            train = False,
                            download = True,
                            transform = transform
)
trainLoader = torch.utils.data.DataLoader(trainDataSet,
                                        batch_size=batchSize
)
testLoader = torch.utils.data.DataLoader(testDataSet,
                                        batch_size=batchSize
)



cnnNet = CNN()
Train(cnnNet, trainLoader, epochs)
Test(cnnNet, testLoader)

torch.save(cnnNet, 'mnistCNN.pt')



