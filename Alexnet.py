import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.datasets import CIFAR10 



class AlexNet(nn.Module):
    def __init__(self, num_classes = 10, dropout = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(256 * 6 * 6, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, num_classes)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
        
transform = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

batch_size = 64

train_data = CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = AlexNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

num_epochs = 30

optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
CE_loss=torch.nn.CrossEntropyLoss()
train_acc = []
test_acc = []
epoch_num = []
for epoch in range(num_epochs):
  epoch_num.append(epoch)
  model.train()
  for data, label in train_loader:
    optimizer.zero_grad()
    data = data.to(device)
    label = label.to(device)
    out = model(data)
    loss = CE_loss(out, label)
    loss.backward()
    optimizer.step()
  model.eval()
  train_accuracy = 0
  train_correct = 0
  train_total = 0
  for train_in, train_target in train_loader:
    with torch.no_grad():
      train_out = model(train_in.to(device))
      train_correct += (torch.argmax(train_out, axis=1) == train_target.to(device)).sum().item()
      train_total += len(train_in)
  train_accuracy = train_correct/train_total
  train_acc.append(train_accuracy)
  test_accuracy = 0
  test_correct = 0
  test_total = 0
  for test_in, test_target in test_loader:
    with torch.no_grad():
      test_out = model(test_in.to(device))
      test_correct += (torch.argmax(test_out, axis=1) == test_target.to(device)).sum().item()
      test_total += len(test_in)
  test_accuracy = test_correct/test_total
  test_acc.append(test_accuracy)
  print("Epoch:    ", epoch, "    Train Accuracy:    ", train_accuracy , "    Test Accuracy:    ", test_accuracy)
torch.save(model.state_dict(),"AlexNet_cifar10.pt")
plt.figure()
plt.plot(epoch_num, train_acc, label = "Train")
plt.plot(epoch_num, test_acc, label = "Test")
plt.title("Accuracy Plot")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig("Accuracy_plots.png")