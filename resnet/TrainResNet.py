import torchvision.models as models
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from data_process import *

ResNet_model=models.resnet50(pretrained=True)
for param in ResNet_model.parameters():
    param.requires_grad=False


ResNet_model.fc=nn.Linear(ResNet_model.fc.in_features,10)
ResNet_model.cuda()

for param in ResNet_model.fc.parameters():
    param.requires_grad=True

optimizer = torch.optim.Adam(ResNet_model.fc.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=19, gamma=0.3)

criterion=F.binary_cross_entropy

train_set=ENSG_Protein_Dataset("train")
trainloader=DataLoader(dataset=train_set,batch_size=64,shuffle=True)

for epoch in range(50):
    running_loss=0.0
    scheduler.step()
    n=0
    for i,data in enumerate(trainloader,0):
        n=n+1
        inputs,labels=data
        inputs=inputs.cuda()
        labels=labels.float()
        outputs=ResNet_model(inputs).cpu()
        outputs=torch.sigmoid(outputs)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        running_loss+=loss.item()
        optimizer.step()
        print('loss:%.3f'%(loss))        


ResNet_model.cpu()
print("fuck you")
torch.save(ResNet_model, 'myResNet50.pth')
