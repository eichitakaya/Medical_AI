import torch.nn as nn
import torchvision

import poolformer

class Net3(nn.Module):
    def __init__(self, size, in_channels, num_classes):
        super(Net3,self).__init__()
        
        self.unit_num = {
            "28":256,
            "56":5184,
            "112": 33856,
            "224": 166464
        }
        
        #畳み込み層
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = 5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride=1, padding=0),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #全結合層
        self.dense = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.unit_num[str(size)], 256), #入力サイズによって変えなければならない, 28 = 256, 56 = 5184, 112 = 33856,224 =166464
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
         
    #順伝播
    def forward(self,x):
         
        out = self.conv_layers(x)
        #Flatten
        out = out.view(out.size(0), -1)
        #全結合層
        out = self.dense(out)
         
        return out
     
    #畳み込み層の出力サイズのチェック
    def check_cnn_size(self, size_check):
        out = self.conv_layers(size_check)
         
        return out


class Net5(nn.Module):
    def __init__(self, size, in_channels, num_classes):        
        super(Net5, self).__init__()
        
        self.unit_num = {
            "28":1024,
            "56":7744,
            "112": 40000,
            "224": 179776
        }
        

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(self.unit_num[str(size)], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, size, in_channels, num_classes):
        super().__init__()

        self.feature = torchvision.models.resnet18(pretrained=False)
        self.feature.conv1.weight = nn.Parameter(self.feature.conv1.weight.sum(dim=1).unsqueeze(in_channels))
        self.fc = nn.Linear(1000, num_classes)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


class ResNet50(nn.Module):
    def __init__(self, size, in_channels, num_classes):
        super().__init__()

        self.feature = torchvision.models.resnet50(pretrained=False)
        self.feature.conv1.weight = nn.Parameter(self.feature.conv1.weight.sum(dim=1).unsqueeze(in_channels))
        self.fc = nn.Linear(1000, num_classes)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


class Poolformer(nn.Module):
    def __init__(
            self, size, in_channels, 
            num_classes,
            in_patch_size=7, 
            in_stride=4, 
            in_pad=2, 
            down_patch_size=3, 
            down_stride=2, 
            down_pad=1, 
            drop_rate=0., 
            drop_path_rate=0., 
            embed_dims=None
                 ):
        super().__init__()
        embed_dims = [64, 128, 320, 512]
        self.feature = poolformer.poolformer_s12()
        self.feature.patch_embed = poolformer.PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=in_channels, embed_dim=embed_dims[0])
        self.fc = nn.Linear(1000, num_classes)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
