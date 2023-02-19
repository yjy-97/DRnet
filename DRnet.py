import pandas as pd
import torch
import math
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from utils import PosCNN, SELayer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


# Data preprocessing
# To facilitate the distinction, we unify the naming rules for the facial expression images imitated by the case and control subjects.
# The names of the facial expressions of the patients in the case group start with an English capital letter D, and the facial expressions of the control group begin with the letter N;
# Named with three digits according to the collection sequence of each subject, such as 001, 002;
class DepDataSet(Dataset):
    def __init__(self, train=True):

        # Please update your data path here
        self.root = r'./dataset/'

        if train:
            self.file_path = os.path.join(self.root, 'train_csv')
            self.label_list = os.listdir(self.file_path)

        else:
            self.file_path = os.path.join(self.root, 'test_csv')
            self.label_list = os.listdir(self.file_path)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        fileName = self.label_list[idx]
        label = self.label_list[idx].split('.')[0][0]
        if label == 'N':
            label = 1
        if label == 'D':
            label = 0
        data = np.array(pd.read_csv(os.path.join(self.file_path, fileName)))[:, 1:]
        data = torch.unsqueeze(torch.tensor(data), dim=0)

        return data.type(torch.float32), label


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1, 1], padding=[0, 1, 0], first=False) -> None:
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=padding[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=padding[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=stride[2], padding=padding[2], bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )

        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# Model structure of DRnet
class DRnet(nn.Module):
    def __init__(self, Bottleneck, num_classes=2):
        super(DRnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # WSM
        self.wsm = nn.Sequential(nn.Linear(2048, 710), nn.Sigmoid())

        # SE
        self.se1 = SELayer(64)
        self.se2 = SELayer(256)
        self.se3 = SELayer(512)
        self.se4 = SELayer(1024)
        self.se5 = SELayer(2048)

        # CVPT
        self.peg = PosCNN(23, 23)

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.se1(out)
        out = self.conv2(out)
        out = self.se2(out)
        out = self.conv3(out)
        out = self.se3(out)
        out = self.conv4(out)
        out = self.se4(out)
        out = self.conv5(out)
        out = self.se5(out)

        out = out.view(out.shape[0], out.shape[1], -1)
        out = self.peg(out, 4, 512)
        out = torch.unsqueeze(out, dim=2)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        wsm = self.wsm(out)
        out = self.fc(out)
        return out, wsm[-1, :]


batch_size = 64
train_dataset = DepDataSet(train=True)
train_data = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True)

test_dataset = DepDataSet(train=False)
valid_data = DataLoader(test_dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DR = DRnet(Bottleneck)
loss_func = nn.NLLLoss()
optimizer = optim.Adam(DR.parameters(), lr=1e-4)


def train_and_valid(model, loss_function, optimizer, epochs=25):
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        # Initialization of WSM
        global w_train
        w_train = torch.ones(1, 710)
        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # WSM: Change the input with weights
            inputs = torch.mul(inputs, w_train)

            # Because the gradient here is cumulative, so remember to clear it every time
            optimizer.zero_grad()
            outputs, weight_train = model(inputs)

            # Modify the weights at the first batch of every 15 epochs
            if i == 0 and epoch % 15 == 0:
                w_train = weight_train.detach()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            all_preds, all_label = [], []
            w_test = torch.ones(1, 710)
            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # WSM: Change the input with weights
                inputs = torch.mul(inputs, w_test)

                outputs, weight_test = model(inputs)

                # Modify the weights at the first batch of every 15 epochs
                if j == 0 and epoch % 15 == 0:
                    w_test = weight_test

                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

            if len(all_preds) == 0:
                all_preds.append(predictions.detach().cpu().numpy())
                all_label.append(labels.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], predictions.detach().cpu().numpy(), axis=0)
                all_label[0] = np.append(all_label[0], labels.detach().cpu().numpy(), axis=0)

            all_preds, all_label = all_preds[0], all_label[0]
            print("F1-Score:{:.4f}".format(f1_score(all_label, all_preds)))
            print("Recall_score:{:.4f}".format(recall_score(all_label, all_preds)))
            print("Precision_score:{:.4f}".format(precision_score(all_label, all_preds)))

        avg_train_loss = train_loss / train_dataset.__len__()
        avg_train_acc = train_acc / train_dataset.__len__()

        avg_valid_loss = valid_loss / test_dataset.__len__()
        avg_valid_acc = valid_acc / test_dataset.__len__()

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Train_Accuracy: {:.4f}%, Test_Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_acc * 100, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

        w_train = weight_train.detach()
    return model, history


num_epochs = 60
trained_model, history = train_and_valid(DR, loss_func, optimizer, num_epochs)

plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('_accuracy_curve.png')
plt.show()
