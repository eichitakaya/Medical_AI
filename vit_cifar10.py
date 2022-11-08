import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from vit import Vit

# ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# トレーニングデータをダウンロード
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=10)

# テストデータをダウンロード
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=True, num_workers=10)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Vit()
model.load_state_dict(torch.load("./pretrained_models/ImageNet/tiny16/best_checkpoint.pth"))
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.to(device)

# 交差エントロピー
criterion = nn.CrossEntropyLoss()
# 確率的勾配降下法
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 誤差逆伝播
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        running_loss += loss.item()
    

    # validation
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    acc = float(correct / len(testset))
    
    print('[epoch %d] train_loss: %.3f, test_acc: %.3f' % (epoch + 1, running_loss / len(trainset), acc))
    running_loss = 0.0


print('Finished Training')

PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)

