import numpy as np

import torch
import torchvision

from PIL import Image
import cv2


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

def resize_batch(batch, size, method):
    
    method_dir = {
        "nearest": Image.NEAREST,
        "box": Image.BOX,
        "bilinear": Image.BILINEAR,
        "hamming": Image.HAMMING,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS
    }
    
    interpolation_method = method_dir[method.lower()] # interpolation methodが大文字が指定されても問題ないように，ここでlowerに統一
    
    # カラーだったらチャンネルの軸を追加する
    if batch.dim() > 3:
        out_batch = torch.zeros(batch.shape[0], batch.shape[1], size, size)
    else:
        out_batch = torch.zeros(batch.shape[0], size, size) # 空のtensorを用意

    for i in range(len(batch)): # バッチサイズの分だけ繰り返す
        img =  torchvision.transforms.functional.to_pil_image(batch[i]) # tensorをimageに変換
        img = img.resize((size, size), interpolation_method) # リサイズ
        tensor = torchvision.transforms.functional.to_tensor(img) # imageをtensorに変換
        out_batch[i] = tensor # out_batchのi番目に格納
    
    return out_batch