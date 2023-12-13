import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from torch.multiprocessing import freeze_support

from load_data import load_data, load_recommend_data

torch.manual_seed(2023)

split_percent = 0.8
batch_size = 128
num_workers = 8
max_epochs = 15
learning_rate = 0.0005

# SPECIFY FILEPATHS BELOW
username = ""
path_to_model = ""
path_to_save_model = ""
path_to_user_data = ""
path_to_recommend_data = ""

num_classes = 9 # 1 if MSE
needs_save = True
noRound = False # True if MSE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, learning_rate, max_epochs):
        super(CNN, self).__init__()

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
                    9
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
                    total_data += labels.size(0)
                    correct_data += (pred == labels).sum().item()

                acc_rate = correct_data / total_data
                # tepoch.set_postfix({"Loss": epoch_loss, "Accuracy": acc_rate})
            print("Loss: ", epoch_loss)
            print("Accuracy: ", acc_rate)

    def predict(self, test_loader):
        with torch.no_grad():
            pred_loss = 0
            total_data = 0
            correct_data = 0

            for j, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)

                if noRound:
                    labels = labels.unsqueeze(1).to(torch.float32)

                output = self.forward(images)

                _, pred = torch.max(output.data, 1)

                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()

            print("Testing accuracy: ", correct_data / total_data)

    def recommend(self, idxs, recommend_test_loader):
        with torch.no_grad():
            pred_labels = {val: [] for val in idxs}

            for j, (images, labels) in enumerate(recommend_test_loader):
                images, labels = images.to(device), labels.to(device)
                output = self.forward(images)

                if noRound:
                    pred_labels[idxs[int(labels.cpu().item())]].append((output.cpu().item()))

                else:
                    _, pred = torch.max(output.data, 1)

                    for i in range(len(pred)):
                        pred_labels[idxs[int(labels[i].cpu().item())]].append((pred[i].cpu().item()))

            for val in pred_labels:
                val_list = pred_labels[val]
                mean = sum(val_list) / len(val_list)

                pred_labels[val] = mean

        print(pred_labels)

    def forward(self, X):
        return self.layers(X)

if __name__ == "__main__":
    freeze_support()

    if needs_save:
        train_data, test_data, train_loader, test_loader = load_data(path_to_user_data, split_percent, batch_size, num_workers)

        if noRound:
            criterion = nn.MSELoss().to(device)

        else:
            criterion = nn.CrossEntropyLoss().to(device)

        model = torch.load(path_to_model)

        new_fc_layer = nn.Linear(
                model.layers[-1].in_features,
                num_classes,
                )
        model.layers[-1] = new_fc_layer
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

        model.fit(train_loader, criterion, optimizer)
        torch.save(model, path_to_save_model)

        model.predict(test_loader)

    else:
        idxs, recommend_test_loader = load_recommend_data(path_to_recommend_data, 1, num_workers)

        model = torch.load(path_to_save_model)
        model.eval()

        model.recommend(idxs, recommend_test_loader)





