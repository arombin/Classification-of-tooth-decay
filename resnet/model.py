#!/usr/bin/env python
# coding: utf-8

# In[50]:

from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
import os
import cv2
from PIL import Image
import numpy as np
import torch
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models.resnet as resnet
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from tqdm.notebook import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

conv1x1=resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock= resnet.BasicBlock

T1 = 100
T2 = 700
af = 3
best_epoch_semi = 0
best_loss_semi = 0
ls = nn.CrossEntropyLoss()

acc_scores = []
unlabel = []
pseudo_label = []

alpha_log = []
test_acc_log = []
test_loss_log = []


# In[51]:


class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=3, zero_init_residual=True):
        super(ResNet, self).__init__()
        self.inplanes = 32 # conv1에서 나올 채널의 차원 -> 이미지넷보다 작은 데이터이므로 32로 조정

        # inputs = 3x224x224 -> 3x128x128로 바뀜
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 32, layers[0], stride=1) # 3 반복
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2) # 4 반복
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2) # 6 반복
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2) # 3 반복
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1): # planes -> 입력되는 채널 수
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: 
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
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
        x = self.fc(x)

        return x
    


# In[52]:


def train_epoch(model,device,dataloader,loss_fn,optimizer):
    train_loss,train_correct=0.0,0
    model.train()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()

    return train_loss,train_correct


# In[53]:


def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:

        images,labels = images.to(device),labels.to(device)
        output = model(images)
        loss=loss_fn(output,labels)
        valid_loss+=loss.item()*images.size(0)
        scores, predictions = torch.max(output.data,1)
        val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct


# In[74]:


def res_train(path, weight_name):
    path1 = path + '/Team1/resnet/data/augmentation' #train data set
    path2 = path + '/Team1/resnet/data/labeled_test' #test data set
    path3 = path + '/Team1/resnet/data/saved_models' #학습된 모델 가중치 저장 경로
    TRAIN_BS = 32 
    TEST_BS = 32
    num_workers = 2 # Thread 숫자 지정 (병렬 처리에 활용할 쓰레드 숫자 지정)

    trans = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                ])


    train_data = ImageFolder(root = path1,
                          transform= trans)
    test_data = ImageFolder(root = path2,
                          transform= trans)
    
    train_loader = DataLoader(train_data,
                          batch_size=TRAIN_BS,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True)
    test_loader = DataLoader(test_data,
                          batch_size=TEST_BS,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 3, True)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    loss_fn = nn.CrossEntropyLoss()
    splits=KFold(n_splits=5,shuffle=True,random_state=42)
    foldperf={}
    
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_data)))):

        print('Fold {}'.format(fold + 1))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        test_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 3, True)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 20

        history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

        for epoch in range(num_epochs):
            train_loss, train_correct=train_epoch(model,device,train_loader,loss_fn,optimizer)
            test_loss, test_correct=valid_epoch(model,device,test_loader,loss_fn)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1,
                                                                                                                 num_epochs,
                                                                                                                 train_loss,
                                                                                                                 test_loss,
                                                                                                                 train_acc,
                                                                                                                 test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        foldperf['fold{}'.format(fold+1)] = history  

    torch.save(model.state_dict(), path3+'/saved_models/'+weight_name) 


# In[75]:


def evaluate(model, test_loader):
    model.eval()
    correct = 0 
    loss = 0
    ls = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            output = model(data)
            predicted = torch.max(output,1)[1]
            correct += (predicted == labels.cuda()).sum()
            loss += ls(output, labels.cuda()).item()

    return (float(correct)/len(test_data)) *100, (loss/len(test_loader))


# In[76]:


def alpha_weight(step):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af


# In[92]:


def semisup_train(model, train_loader, unlabeled_loader, test_loader, path5):

    
    mi_loss = np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    EPOCHS = 40
    
    step = 95
    
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
            
            # Forward Pass to get the pseudo labels
            x_unlabeled = x_unlabeled[0].cuda()
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()
            
            
            """ ONLY FOR VISUALIZATION"""
            if (batch_idx < 3) and (epoch % 10 == 0):
                unlabel.append(x_unlabeled.cpu())
                pseudo_label.append(pseudo_labeled.cpu())
            """ ********************** """
            
            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step) * ls(output, pseudo_labeled)   
            
            # Backpropogate
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
            
            
            # For every 50 batches train one epoch on labeled data 
            if batch_idx % 50 == 0:
                
                # Normal training procedure
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                    output = model(X_batch)
                    labeled_loss = ls(output, y_batch)

                    optimizer.zero_grad()
                    labeled_loss.backward()
                    optimizer.step()
                
                # Now we increment step by 1
                step += 1
                

        test_acc, test_loss =evaluate(model, test_loader)
        print('Epoch: {} : Alpha Weight : {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, alpha_weight(step), test_acc, test_loss))
        
        if test_loss < mi_loss:
          print(f'Saving Model! [INFO] test_loss has been improved from {mi_loss:.5f} to {test_loss:.5f}. ')
          mi_loss = test_loss
          torch.save(model.state_dict(), path5)
          best_epoch_semi = epoch
          best_loss_semi = mi_loss

        """ LOGGING VALUES """
        alpha_log.append(alpha_weight(step))
        test_acc_log.append(test_acc/100)
        test_loss_log.append(test_loss)
        """ ************** """
        model.train()


# In[95]:


def semi_train(path, weight_name1, weight_name2):
    path1 = path+'/Team1/resnet/data/augmentation2'
    path2 = path+'/Team1/resnet/data/unlabeled'
    path3 = path+'/Team1/resnet/data/labeled_val'
    path4 = path+'/Team1/resnet/saved_models/'+weight_name1
    path5 = path+'/Team1/resnet/saved_models/'+weight_name2
    TRAIN_BS = 32
    TEST_BS = 32
    UNLABELED_BS = 32
    num_workers = 2 # Thread 숫자 지정 (병렬 처리에 활용할 쓰레드 숫자 지정)

    trans = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                ])

    train_data = ImageFolder(root = path1,
                          transform= trans)
    val_data = ImageFolder(root = path3,
                          transform= trans)
    unlabelset = ImageFolder(root = path2,
                          transform= trans)
    
    train_loader = DataLoader(train_data,
                          batch_size=TRAIN_BS,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True)
    val_loader = DataLoader(val_data,
                          batch_size=TEST_BS,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True)
    unlabeled_loader = DataLoader(unlabelset,
                          batch_size=UNLABELED_BS,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 3, True)
    model.load_state_dict(torch.load(path4))
    model.to(device)
    
    
    semisup_train(model, train_loader, unlabeled_loader, val_loader, path5)

    
# In[96]:


def confusion_m(path, weight_name):
    path1 = path+'/Team1/resnet/data/labeled_test'
    path2 = path+'/Team1/resnet/saved_models/'+weight_name
    
    os.chdir(path1+'/filling') 
    files = os.listdir(path1)
    os.chdir(path1+'/no_treatment') 
    files2 = os.listdir(path1)
    os.chdir(path1+'/root_canal') 
    files3 = os.listdir(path1)
    lenfile=len(files)+len(files2)+len(files3)
    
    trans = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                ])

    test_data = ImageFolder(root = path1,
                          transform= trans)

    test_loader = DataLoader(test_data,
                              batch_size=lenfile,
                              shuffle=False,
                              num_workers=2,
                              drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 3, True)
    model.to(device)
    model.load_state_dict(torch.load(path2))
    
    os.chdir(path)
    
    y_pred = []
    y_true = []

    for inputs, labels in test_loader:
            inputs = inputs.cuda()
            output = model(inputs) 

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) 

            labels = labels.data.cpu().numpy()
            y_true.extend(labels) 

    classes = ('filling','no treatment','root canal')

    confusionmatrix = confusion_matrix(y_true, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionmatrix, display_labels = [0,1,2])
    cm_display.plot()
    plt.show()
    

# In[97]:
    
    
def predictList(path, weight_name, person):

    os.chdir(path+'/Team1/resnet/data/person/'+person) 
    files = os.listdir(path+'/Team1/resnet/data/person/'+person)
    
    trans = transforms.Compose([transforms.Resize((128, 128)),
                                transforms.ToTensor(),
                                ])
    list_data = ImageFolder(root = path+'/Team1/resnet/data/person', 
                          transform= trans)
    list_loader = DataLoader(list_data,
                              batch_size=len(files),
                              shuffle=False,
                              num_workers=2,
                              drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 3, True)
    model.to(device)
    model.load_state_dict(torch.load(path+'/Team1/resnet/saved_models/'+weight_name))
    
    path1 = path+'/Team1/resnet/data/person/'+person
    os.chdir(path1) 
    files = os.listdir(path1)
    
    predictlist = [[0 for col in range(2)] for row in range(len(files))]
    newsize = (128, 128)
    convert_tensor = transforms.ToTensor()
    i=0
    for data, label in list_loader:
        data = data.cuda()
        output = model(data)
        predicted = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        for i in range(0, len(files)):
          predictlist[i][0]= files[i]
          predictlist[i][1]=predicted[i]
    
    predictlist.sort()
    a=[]
    b=[]
    c=[]
    
    for i in range(len(predictlist)):  # list sort 함수
        if predictlist[i][0][-6] == '_':
            a.append(predictlist[i])
        else:
            b.append(predictlist[i])
    
    c = a+b
    
    return c