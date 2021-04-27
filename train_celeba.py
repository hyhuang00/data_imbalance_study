import os
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

def makedir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, npy_path, img_dir, label_col, add_label_col, transform=None):

        npy = np.load(npy_path, allow_pickle=True)
        self.img_dir = img_dir
        self.npy_path = npy_path
        self.img_names = npy[:,0]
        self.y = ((npy[:,label_col] + 1) / 2).astype(int)
        self.y_add = ((npy[:,add_label_col] + 1) / 2).astype(int)
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

# Note that transforms.ToTensor()
# already divides pixels by 255. internally
def arg_prep():
    '''
    A helper function that prepares the command line
    arguments for the python script.
    '''
    additional_args = [
        {
        'flag': '--index',
        'help': 'Index of training set.',
        'action': 'store',
        'type': int,
        'dest': 'index',
        'required': True
        },
    ]
    return additional_args


def parse_additional_args(description="", additional_args=[], use_default=True):
    '''
    Parse command line arguments for the script.
    Input:
        description: a string. Description of the program
        additional_args: a list of dicts. Additional arguments
                         of the script.
    Output:
        args: a Namespace. Contains the parsed command line arguments.
    '''
    parser = argparse.ArgumentParser(description=description)
    for arg_dict in additional_args:
        parser.add_argument(arg_dict['flag'],
                            help=arg_dict['help'],
                            action=arg_dict['action'],
                            type=arg_dict['type'],
                            dest=arg_dict['dest'],
                            required=arg_dict['required'])
    args = parser.parse_args()
    return args


##########################
### MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas, x

def resnet18(num_classes):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2, 2, 2, 2],
                   num_classes=NUM_CLASSES,
                   grayscale=GRAYSCALE)
    return model

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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

##########################
### COST AND OPTIMIZER
##########################
def main(style='smile_gender', index=0):

    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                        transforms.Resize((128, 128)),
                                        transforms.ToTensor()])
    if style == 'smile_gender':
        add_label_col = 2
    else:
        add_label_col = 3

    train_dataset = CelebaDataset(npy_path=f'./CelebA/{style}_npys/train_{index}.npy',
                                img_dir='./CelebA/img_align_celeba/',
                                label_col=1, # smile
                                add_label_col=add_label_col, # 2: gender, 3: open mouth
                                transform=custom_transform)

    test_datasets = []
    for i in range(3):
        test_dataset = CelebaDataset(npy_path=f'./CelebA/{style}_npys/test_{i}.npy',
                                    img_dir='./CelebA/img_align_celeba/',
                                    label_col=1, # smile
                                    add_label_col=add_label_col, # 2: gender, 3: open mouth
                                    transform=custom_transform)
        test_datasets.append(test_dataset)


    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=4)
    
    eval_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=4)

    test_loaders = []
    for i in range(3):
        test_loader = DataLoader(dataset=test_datasets[i],
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=4)
        test_loaders.append(test_loader)
    torch.manual_seed(0)

    torch.manual_seed(RANDOM_SEED)
    model = resnet18(NUM_CLASSES)

    #### DATA PARALLEL START ####
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    #### DATA PARALLEL END ####


    model.to(DEVICE)

    cost_fn = torch.nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
                
            ### FORWARD AND BACK PROP
            logits, probas, _ = model(features)
            cost = cost_fn(logits, targets)
            optimizer.zero_grad()
            
            cost.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1, NUM_EPOCHS, batch_idx, 
                        len(train_loader), cost))

        model.eval()
        if (epoch + 1) % 10 == 0:
            with torch.set_grad_enabled(False):  # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%% ' % (
                    epoch+1, NUM_EPOCHS,
                    compute_accuracy(model, eval_loader, device=DEVICE, 
                    savex=f'./penultimates/{style}_{index}_{epoch+1}.npy')))
            if (epoch + 1) >= 60:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, filename=os.path.join("./CelebA/weights", f'checkpoint_{epoch:04d}.pth.tar'))

        else:
            with torch.set_grad_enabled(False):  # save memory during inference
                print('Epoch: %03d/%03d | Train: %.3f%% ' % (
                    epoch+1, NUM_EPOCHS,
                    compute_accuracy(model, eval_loader, device=DEVICE)))
            
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    model.eval()
    with torch.set_grad_enabled(False): # save memory during inference
        makedir(f'./pred_labels/{style}_long/')
        for i in range(3):
            savestr = f'./pred_labels/{style}_long/{index}_{i}.npy'
            print(f'Test accuracy: {index}/{i} : {(compute_accuracy(model, test_loaders[i], device=DEVICE,save=savestr)):.2f}' )

if __name__ == '__main__':
    ##########################
    ### SETTINGS
    ##########################

    # Hyperparameters
    RANDOM_SEED = 202
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100

    # Architecture
    NUM_FEATURES = 128*128
    NUM_CLASSES = 2
    BATCH_SIZE = 256*torch.cuda.device_count()
    DEVICE = 'cuda:0' # default GPU device
    GRAYSCALE = False
    # style = 'smile_open'
    style = 'smile_gender'
    print(style)
    for ind in [0]:
        main(style, index=ind)
