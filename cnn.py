%matplotlib inline
import torch
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def plot_kernel(model):
    model_weights = model.state_dict()
    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(model_weights['conv1.weight']):
    #print(filt[0, :, :])
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt[0, :, :], cmap="gray")
        plt.axis('off')
    
    plt.show()

def plot_kernel_output(model,images):
    fig1 = plt.figure()
    plt.figure(figsize=(1,1))
    
    img_normalized = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
    plt.imshow(img_normalized.numpy().transpose(1,2,0))
    plt.show()
    output = model.conv1(images)
    layer_1 = output[0, :, :, :]
    layer_1 = layer_1.data

    fig = plt.figure()
    plt.figure(figsize=(10,10))
    for idx, filt  in enumerate(layer_1):
        if idx >= 32: continue
        plt.subplot(4,8, idx + 1)
        plt.imshow(filt, cmap="gray")
        plt.axis('off')
    plt.show()

def test_accuracy(net, dataloader):
  ########TESTING PHASE###########
  
    #check accuracy on whole test set
    correct = 0
    total = 0
    net.eval() #important for deactivating dropout and correctly use batchnorm accumulated statistics
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: %d %%' % (
    accuracy))
    return accuracy

    
n_classes = 100 
#function to define the convolutional network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 512, kernel_size=5, stride=2, padding=0)
        self.conv1_bn = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_final = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv_final_bn = nn.BatchNorm2d(1024)        
        self.fc1 = nn.Linear(1024 * 4 * 4, 4096)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4096, n_classes) #last FC for classification 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = F.relu(self.pool(self.conv_final_bn(self.conv_final(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x) #Dropout
        x = self.fc2(x)
        return x

""""      
class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1, affine=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.1, affine=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.1, affine=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512, momentum=0.1, affine=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.m = nn.AvgPool2d(3, stride=1)
             
        self.fc1 = nn.Linear(512*4*4, 4096)
        self.dropout = nn.Dropout(0.8)
        self.fc2 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        #Layer 1
        #1-Basic block 0
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.bn1(self.conv2(x))
        #1-Basic block 1
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.bn1(self.conv2(x))
        #Layer 2
        #2-Basic block 0
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.bn2(self.conv4(x))
        #2-Basick block 1
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.bn2(self.conv4(x))
        #Layer 3
        #3-Basic block 0
        x = F.relu(self.bn3(self.conv5(x)))
        x = self.bn3(self.conv6(x))
        #3-Basic block 1
        x = F.relu(self.bn3(self.conv6(x)))
        x = self.bn3(self.conv6(x))
        #Layer 4
        #4-Basic block 0
        x = F.relu(self.bn4(self.conv7(x)))
        x = self.bn4(self.conv8(x))
        #4-Basic block 1
        x = F.relu(self.bn4(self.conv8(x)))
        x = self.bn4(self.conv8(x))
        #AVG Pool 2s
        #x = self.m(x)
        #Fully connected layer
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #x = dropout(x)
        x = self.fc2(x)
        return x
"""""

      
#transform are heavily used to do simple and complex transformation and data augmentation
transform_train = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     #transforms.Resize((40, 40)), #crop
     #transforms.RandomCrop((32, 32)), #crop
     transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose(
    [
     transforms.Resize((32,32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=4,drop_last=True)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=4,drop_last=True)


dataiter = iter(trainloader)

###Show images:
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
###


###CNN
net = CNN()
####


###Kernel:
#print("####plotting kernels of conv1 layer:####")
#plot_kernel(net)
###

net = net.cuda()

criterion = nn.CrossEntropyLoss().cuda() #it already does softmax computation for use!
#optimizer = optim.Adam(net.parameters(), lr=0.001)
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)

###Kernel:
#print("####plotting output of conv1 layer:#####")
#plot_kernel_output(net,images)  
###

########TRAINING PHASE###########
n_loss_print = len(trainloader)  #print every epoch, smaller numbers will print loss more often!

losses=[]
accuracy = []

n_epochs = 30
for epoch in range(n_epochs):  # loop over the dataset multiple times
    net.train() #important for activating dropout and correctly train batchnorm
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs and cast them into cuda wrapper
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % n_loss_print == (n_loss_print -1):    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / n_loss_print))
            losses.append(running_loss / n_loss_print)
            running_loss = 0.0
    accuracy.append(test_accuracy(net,testloader))

print('Finished Training')

plt.title('Training loss & accuracy curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy / Loss')
plt.plot(range(n_epochs),accuracy, label='Accuracy')
plt.plot(range(n_epochs),losses, label='Loss')
plt.legend()