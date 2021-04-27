import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, npy_path, img_dir, label_col, add_label_col, transform=None):

        npy = np.load(npy_path, allow_pickle=True)
        self.img_dir = img_dir
        self.npy_path = npy_path
        self.img_names = npy[:, 0]
        self.y = ((npy[:, label_col] + 1) / 2).astype(int)
        self.y_add = ((npy[:, add_label_col] + 1) / 2).astype(int)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]


def makedir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

##########################
# SETTINGS
##########################

# Architecture
NUM_FEATURES = 784
NUM_CLASSES = 2
BATCH_SIZE = 128
DEVICE = 'cuda:0'  # default GPU device
GRAYSCALE = False


def Cifar10Subset(data_dir, index):
    train_indices = np.load(
        f'./cifar10/training_{index}.npy', allow_pickle=True)
    test_indices = np.load(
        './cifar10/testing.npy', allow_pickle=True)

    train_dataset = Subset(datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transforms.ToTensor()),
                           train_indices)
    test_dataset = Subset(datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transforms.ToTensor()),
                          test_indices)
    train_dataset.dataset.targets = np.reshape(
        train_dataset.dataset.targets, (-1))
    train_dataset.dataset.targets[train_dataset.dataset.targets == 3] = 0
    train_dataset.dataset.targets[train_dataset.dataset.targets == 5] = 1
    test_dataset.dataset.targets = np.reshape(
        test_dataset.dataset.targets, (-1))
    test_dataset.dataset.targets[test_dataset.dataset.targets == 3] = 0
    test_dataset.dataset.targets[test_dataset.dataset.targets == 5] = 1

    return train_dataset, test_dataset

##########################
# MODEL
##########################


class VGG16(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(VGG16, self).__init__()

        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      # (1(32-1)- 32 + 3)/2 = 1
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )
        self.block_4 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                         stride=(2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
        )

        self.last_layer = nn.Linear(4096, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()


    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        logits = self.last_layer(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas, x


def compute_accuracy(model, data_loader, device, save=None, savex=None):
    correct_pred, num_examples = 0, 0
    pred_labels = []
    penultimate = []
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas, x = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        if save is not None:
            pred_labels.append(predicted_labels.cpu().numpy())
        if savex is not None:
            penultimate.append(x.cpu().numpy())
    if save is not None:
        pred_labels = np.array(pred_labels)
        np.save(save, pred_labels)
    if savex is not None:
        all_repr = penultimate[0]
        for j in range(1, len(penultimate)):
            all_repr = np.concatenate((all_repr, penultimate[j]))
        np.save(savex, all_repr)
    return correct_pred.float()/num_examples * 100


def main(index):
    # DATASET
    train_dataset, test_dataset = Cifar10Subset(
        "./cifar10", index)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)

    eval_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)

    # COST AND OPTIMIZER
    torch.manual_seed(RANDOM_SEED)
    model = VGG16(num_features=NUM_FEATURES,
                  num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    cost_fn = torch.nn.CrossEntropyLoss()

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # FORWARD AND BACK PROP
            logits, probas, _ = model(features)
            cost = cost_fn(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                      % (epoch+1, NUM_EPOCHS, batch_idx,
                         len(train_loader), cost))

        model.eval()
        if (epoch + 1) % 10 == 0:
            with torch.set_grad_enabled(False):  # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%% ' % (
                    epoch+1, NUM_EPOCHS,
                    compute_accuracy(model, eval_loader, device=DEVICE, 
                    savex=f'./penultimates/{index}_{epoch+1}.npy')))
        else:
            with torch.set_grad_enabled(False):  # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%% ' % (
                    epoch+1, NUM_EPOCHS,
                    compute_accuracy(model, eval_loader, device=DEVICE)))

        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    makedir(f'./pred_labels/cifar10_long/')
    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference
        print(f'Case {index}:')
        print('Test accuracy : %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE,
                                                           save=f'./pred_labels/cifar10_long/{index}.npy')))


if __name__ == '__main__':
    # Hyperparameters
    RANDOM_SEED = 20200200
    LEARNING_RATE = 0.00002
    NUM_EPOCHS = 250

    for index in range(9):
        main(index)
