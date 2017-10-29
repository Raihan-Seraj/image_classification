import numpy as np 
import scipy.misc 
import csv
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
def visualise_data():
    data=[]
    reader=csv.reader(open('dataset/train_x.csv'),delimiter=",")
    count=0
    for row in reader:
        data.append(row)
        count+=1
        if count>1:
            break

    print("done")
    z= np.reshape(data, (-1,64, 64))
    z=z.astype(float)

    scipy.misc.imshow(z[0,:,:])

##Setting up the hyper parameters
num epochs=5
batch_size=100
learning_rate=0.001

##Defining the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cnn=CNN()

##Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(cnn.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
