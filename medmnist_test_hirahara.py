from tqdm import tqdm
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import sys
from PIL import Image

from MedMNIST import MedMNIST

# For poolformer
import poolformer


# ---------------------------------------------------------
# Configure
# ---------------------------------------------------------

DATA_NO = 0     # 0: ChestMNIST 1: PneumoniaMNIST 
                # 2: RetinaMNIST 3: BreastMNIST


class Net3(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net3,self).__init__()
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
            nn.Linear(64*51*51, 256), #入力サイズによって変えなければならない, 28 = 256, 56 = 5184, 112 = 33856,224 =166464
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
    def __init__(self, in_channels, num_classes):
        super(Net5, self).__init__()

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
            nn.Linear(64 * 53 * 53, 128),
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
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.feature = torchvision.models.resnet18(pretrained=False)
        self.feature.conv1.weight = nn.Parameter(self.feature.conv1.weight.sum(dim=1).unsqueeze(in_channels))
        self.fc = nn.Linear(1000, num_classes)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h


class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes):
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
            self, in_channels, 
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


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    if ( image.shape[2]==1 ):
        image = image.reshape(image.shape[0], image.shape[1])
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        new_image = new_image.reshape(new_image.shape[0], new_image.shape[1], 1)
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


def ResizeDataset(dataset, size, method, max_data_num, n_channels, n_classes):
    if ( method=="NEAREST" ):
        pil_method = Image.NEAREST
    elif ( method=="BOX" ):
        pil_method = Image.BOX
    elif ( method=="BILINEAR" ):
        pil_method = Image.BILINEAR
    elif ( method=="HAMMING" ):
        pil_method = Image.HAMMING
    elif ( method=="BICUBIC" ):
        pil_method = Image.BICUBIC
    elif ( method=="LANCZOS" ):
        pil_method = Image.LANCZOS

    #num = max_data_num
    #if ( max_data_num>len(dataset) ):
    #    num = len(dataset)
    num = len(dataset)

    images = np.empty((num, n_channels, size, size), dtype=np.uint8)
    labels = np.empty((num, n_classes), dtype=np.uint8)
    for i in range(0, len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]
        x = np.asarray(x)
        x = np.transpose(x , [1,2,0])
        x = x * 255
        x = np.uint8(x)
        #print(x.shape)
        #print(y)
        x = cv2pil(x)
        x = x.resize((size, size), pil_method)
        x = pil2cv(x)
        x = np.transpose(x , [2,0,1])
        images[i] = np.copy(x)
        labels[i] = np.copy(y)
        #if ( i==max_data_num ):
        #    break


    images = torch.tensor(images, dtype=torch.uint8)
    labels = torch.tensor(labels, dtype=torch.uint8)
    dataset = torch.utils.data.TensorDataset(images, labels)
    return(dataset)


def test(model, data_loader, split, task, data_flag, device):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    model = model.to(device)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, dtype=torch.float32)
            outputs = model(inputs)
            outputs = outputs.to("cpu")

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

    return(metrics)


def main():
    data_flag = sys.argv[1]
    SIZE = int(sys.argv[2])
    METHOD = sys.argv[3]
    MODEL = sys.argv[4]
    NUM_EPOCHS = int(sys.argv[5])
    BATCH_SIZE = int(sys.argv[6])
    

    # data_flag = 'breastmnist'
    download = True

    lr = 0.001

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    #data_transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[.5], std=[.5])
    #])
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    train_dataset = ResizeDataset(train_dataset, SIZE, METHOD, len(train_dataset), n_channels, n_classes)

    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    val_dataset = ResizeDataset(val_dataset, SIZE, METHOD, len(val_dataset), n_channels, n_classes)
    
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    test_dataset = ResizeDataset(test_dataset, SIZE, METHOD, len(test_dataset), n_channels, n_classes)

    #pil_dataset = DataClass(split='train', download=download)
    

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    #model = Net5(in_channels=n_channels, num_classes=n_classes)
    #model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    #model = torchvision.models.resnet18(num_classes=n_classes, pretrained=False)
    if ( MODEL=="net3" ):
        model = Net3(in_channels=n_channels, num_classes=n_classes)
    elif ( MODEL=="net5" ):
        model = Net5(in_channels=n_channels, num_classes=n_classes)
    elif ( MODEL=="resnet18" ):
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif ( MODEL=="resnet50" ):
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    elif ( MODEL=="poolformer" ):
        model = Poolformer(in_channels=n_channels, num_classes=n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
        
    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_auc = 0
    best_model = copy.deepcopy(model)
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0

        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            inputs = inputs.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to("cpu")

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
        (auc, acc) = test(model, val_loader, "val", task, data_flag, device)
        print("val auc:%.3f  acc:%.3f" % (auc, acc))
        if ( best_auc<auc ):
            best_auc = auc
            best_model = copy.deepcopy(model)

    best_model = best_model.to("cpu")

    #dataloader = train_loader_at_eval
    #test(model, dataloader, "train", task, data_flag)
    dataloader = test_loader
    (auc, acc) = test(best_model, dataloader, "test", task, data_flag, device)
    print('%s  auc:%.3f  acc:%.3f' % ("test", auc, acc))
    return(0)


if __name__=="__main__":
    main()
