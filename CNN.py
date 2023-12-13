from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import freeze_support

from load_data import load_data

torch.manual_seed(2023)

split_percent = 0.8
batch_size = 128
num_workers = 8
max_epochs = 15
learning_rate = 0.0005

# SPECIFY FILEPATHS BELOW
path_to_data = ""
path_to_test = ""
path_to_model = ""

needs_save = True
noRound = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, learning_rate, max_epochs):
        super(CNN, self).__init__()

        if noRound:
            output_size = 1

        else:
            output_size = 3

        self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels = 3,
                    out_channels = 8,
                    kernel_size = [3, 3],
                    stride = 1,
                    padding = 0,
                    bias = True,
                    ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = 8,
                    out_channels = 12,
                    kernel_size = [3, 3],
                    stride = 1,
                    padding = 0,
                    bias = True,
                    ),
                nn.MaxPool2d(
                    kernel_size = [2, 2],
                    stride = 2,
                    ),
                nn.Dropout(
                    p = 0.5,
                    ),
                nn.Conv2d(
                    in_channels = 12,
                    out_channels = 16,
                    kernel_size = [3, 3],
                    stride = 1,
                    padding = 0,
                    bias = True,
                    ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = 16,
                    out_channels = 20,
                    kernel_size = [3, 3],
                    stride = 1,
                    padding = 0,
                    bias = True,
                    ),
                nn.MaxPool2d(
                    kernel_size = [2, 2],
                    stride = 2,
                    ),
                nn.Dropout(
                    p = 0.5,
                    ),
                nn.Flatten(),
                nn.Linear(
                    64680,
                    3000,
                    ),
                nn.ReLU(),
                nn.Dropout(
                    p = 0.5,
                    ),
                nn.Linear(
                    3000,
                    50
                    ),
                nn.ReLU(),
                nn.Dropout(
                    p = 0.5,
                    ),
                nn.Linear(
                    50,
                    output_size
                    ),
                )

        self.eta = learning_rate
        self.max_epochs = max_epochs

    def fit(self, train_loader, criterion, optimizer):
        steps = 1

        for i in range(1, self.max_epochs):
            with tqdm(train_loader, unit = "batch") as tepoch:

                tepoch.set_description(f"Epoch {i}")
                epoch_loss = 0
                total_data = 0
                correct_data = 0

                for j, (images, labels) in enumerate(tepoch):
                    images, labels = images.to(device), labels.to(device)

                    if noRound:
                        labels = labels.unsqueeze(1).to(torch.float32)

                    output = self.forward(images)

                    loss = criterion(output, labels)
                    loss.backward()

                    if (j + 1) % steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    # optimizer.step()
                    # optimizer.zero_grad()

                    items = len(train_loader)
                    epoch_loss += loss.item() / items
                    _, pred = torch.max(output.data, 1)
                    if noRound:
                        pred = np.round(pred.cpu())
                        labels = labels.cpu()

                    total_data += labels.size(0)
                    correct_data += (pred == labels).sum().item()
                acc_rate = correct_data / total_data
            print("Loss: ", epoch_loss)
            print("Accuracy: ", acc_rate)

    def predict(self, test_loader, criterion):
        with torch.no_grad():
            pred_loss = 0
            total_data = 0
            correct_data = 0
            for j, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)

                if noRound:
                    labels = labels.unsqueeze(1).to(torch.float32)

                output = self.forward(images)
                loss = criterion(output, labels)

                items = len(test_loader)
                pred_loss += loss.item() / items

                _, pred = torch.max(output.data, 1)

                if noRound:
                    pred = np.round(pred.cpu())
                    labels = labels.cpu()

                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()

            print("Testing accuracy: ", correct_data / total_data)

    def forward(self, X):
        return self.layers(X)

if __name__ == "__main__":
    freeze_support()
    train_data, unused_test_data, train_loader, unused_test_loader = load_data(path_to_data, split_percent, batch_size, num_workers)
    unused_train_data, test_data, unused_train_loader, test_loader = load_data(path_to_test, 0.2, batch_size, num_workers)

    model = CNN(learning_rate, max_epochs)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    if noRound:
        criterion = nn.MSELoss().to(device)

    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if needs_save:
        model.fit(train_loader, criterion, optimizer)
        torch.save(model, path_to_model)

    else:
        model = torch.load(path_to_model)
        model.eval()

    model.predict(test_loader, criterion)

