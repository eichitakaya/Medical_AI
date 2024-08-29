import torch
import torch.nn as nn
import torch.nn.functional as F
import networks
from MedMNIST import MedMNIST
from hirahara_utils import cv2pil, pil2cv, resize_batch

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dataset", type=str)
parser.add_argument("interpolation", type=str)
parser.add_argument("size", type=int)
parser.add_argument("network", type=str)
parser.add_argument("epoch", type=int)
parser.add_argument("batchsize", type=int)

args = parser.parse_args()

#----------------------------------------------------------
# ハイパーパラメータなどの設定値
num_epochs = args.epoch         # 学習を繰り返す回数
num_batch = args.batchsize      # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率
image_size = args.size     # 画像の1辺のサイズ

# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成

# MedMNISTデータの取得（下記のリストから選択可能）
'''
"adrenal3d", "fracture3d", "organ3d",
"synapse3d", "blood", "nodule3d",
"organ", "tissue", "breast", "oct", "path",
"vessel3d", "chest", "organa",
"pneumonia", "derma", "organc", "retina"
'''

# 学習用
train_dataset = MedMNIST(
    name = args.dataset,
    traintestval = "train"
    )

# 評価用
test_dataset = MedMNIST(
    name = args.dataset,
    traintestval = "test"
    )

# データセットに応じて下記は変更される
n_channels = train_dataset.n_channels
n_classes = train_dataset.n_class

# データローダー
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,     
    batch_size = num_batch,
    shuffle = True)

#----------------------------------------------------------
# ニューラルネットワークモデルの定義
models = {
    "net3": networks.Net3,
    "net5": networks.Net5,
    "resnet18": networks.ResNet18,
    "resnet50": networks.ResNet50,
    "poolformer": networks.Poolformer
}

#----------------------------------------------------------

# 使用するモデルを指定
model = models[args.network](image_size, n_channels, n_classes).to(device)

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss() 

#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum = 0

    for inputs, labels in train_dataloader:
        # ここでリサイズを施す
        inputs = resize_batch(inputs, image_size, args.interpolation)
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # optimizerを初期化
        optimizer.zero_grad()

        # ニューラルネットワークの処理を行う
        # カラーであればunsqueezeしない
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        
        # 損失(出力とラベルとの誤差)の計算
        loss = criterion(outputs, labels)
        loss_sum += loss

        # 勾配の計算
        loss.backward()

        # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # モデルの重みの保存
    #torch.save(model.state_dict(), 'model_weights.pth')

#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        
        # ここでリサイズを施す
        inputs = resize_batch(inputs, image_size, args.interpolation)
        
        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        # カラーであればunsqueezeしない
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")