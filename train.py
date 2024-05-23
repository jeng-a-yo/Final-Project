import os
from os import walk
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
# import wandb

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from myModels import *

np.random.seed(42)
torch.manual_seed(42)

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.001,
#     "architecture": "CNN",
#     "dataset": [{"dataset": "NumberDataSet"}, {"dataset": "EnglishDataSet"}, {"dataset": "SymbolDataSet"}],
#     "epochs": 30,
#     }
# )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataDir = ["NumberDataSet", "EnglishDataSet", "SymbolDataSet"]
modelName = ["NumberModel", "EnglishModel", "SymbolModel"]
batchSize = 64
epochs = 10
learningRate = 0.001
momentum = 0.9


def MeasureTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        print(f"[Info] Spand Time: {round(time.time() - startTime, 4)} seconds")
        return
    return wrapper


def Train(model, trainLoader, optimizer, criterion, epochs):

    model.train()
    trainAcc, trainLoss = [], []

    for epoch in range(1, epochs+1):
        train_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Training loop
        progressBar = tqdm(enumerate(trainLoader), total=len(trainLoader), desc=f"Epoch {epoch}")
        for batchIdx, (data, target) in progressBar:
            optimizer.zero_grad()
            predict = model(data)
            loss = criterion(predict, target)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            _, predicted = torch.max(predict, 1)
            correct_predictions += (predicted == target).sum().item()
            total_predictions += target.size(0)

            progressBar.set_postfix({'loss': round(loss.item(), 6)})
        
        trainLoss.append(train_running_loss / len(trainLoader))
        trainAcc.append(correct_predictions / total_predictions)

        print(f"[Info] Epoch {epoch}: Training Loss: {trainLoss[-1]}, Training Accuracy: {trainAcc[-1]}")

    print("[Info] Training completed")

    return trainAcc, trainLoss



def Test(model, testLoder):

    model.eval()

    correctCnt = 0
    with torch.no_grad():
        progress_bar = tqdm(testLoder, desc="Testing")
        for data, target in progress_bar:
            predict = model(data)
            answer = predict.argmax(dim=1, keepdim=True)
            correctCnt += answer.eq(target.view_as(answer)).sum().item()
            progress_bar.set_postfix({'accuracy': int(100 * correctCnt / len(testLoder.dataset))})
    
    print(f"[Info] Test Results: Accuracy: {round(100 * correctCnt / len(testLoder.dataset), 2)}%")




@MeasureTime
def main():

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    for i in range(len(dataDir)):
        
        st = time.time()

        # Load the dataset
        dataset = datasets.ImageFolder(root=dataDir[i], transform=transform)

        trainSize = int(0.8 * len(dataset))
        testSize = len(dataset) - trainSize

        trainSet, testSet = random_split(dataset, [trainSize, testSize])

        trainLoader = DataLoader(trainSet, batch_size=batchSize, shuffle=True)
        testLoader = DataLoader(testSet, batch_size=batchSize, shuffle=False)

        # Build the model

        model = PaperCNN(in_channels=1, num_classes=len(os.listdir(dataDir[i]))).to(device)

        # Define loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentum)

        # Train the model
        trainAcc, trainLoss = Train(model, trainLoader, optimizer, criterion, epochs)

        # Evaluate the model
        Test(model, testLoader)

        # Sava the model
        torch.save(model.state_dict(), f'{modelName[i]}.pth')


        # make graph
        plt.figure()
        # loss
        plt.subplot(2,1,1)
        plt.plot(trainLoss, label='train loss')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss')
        # accuracy
        plt.subplot(2,1,2)
        plt.plot(trainAcc, label='train acc')
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        # save figure
        plt.tight_layout()
        plt.savefig(f'{modelName[i]}Graph.png')


        print(f"[Info] Spand Time: {round(time.time() - st, 4)} seconds")
        print("================================================================")


main()

