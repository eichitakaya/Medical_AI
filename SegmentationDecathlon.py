import glob
import torch
import numpy as np
import nibabel as nib
from PIL import Image

class SegmentationDecathlon(torch.utils.data.Dataset):
    def __init__(self, name, traintest="train"):
        super().__init__()
        """

        """
        assert traintest.lower() == "train" or traintest.lower() == "test", "traintest must be 'train' or 'test'."

        # (データセット名，z軸のインデックス, シリーズ数)を保持
        dataset_dir = {
            "brain": ("Task01_BrainTumour", 2, 4),
            "heart": ("Task02_Heart", 2, 1),
            "liver": ("Task03_Liver", 2, 1),
            "hippocampus": ("Task04_Hippocampus", 1, 1),
            "prostate": ("Task05_Prostate", 2, 2),
            "lung": ("Task06_Lung", 2, 1),
            "pancreas": ("Task07_Pancreas", 2, 1),
            "hepaticvessel": ("Task08_HepaticVessel", 2, 1),
            "spleen": ("Task09_Spleen", 2, 1),
            "colon": ("Task10_Colon", 2, 1)
        }
        assert name.lower() in dataset_dir, "The Spefified dataset {} does not exist.".format(name)
        
        # dataフォルダのあるディレクトリの絶対パス(initの引数にしてもいいかも？)
        datafolder_path = "/takaya_workspace/data/decathlon/"

        self.dataset_path = datafolder_path + dataset_dir[name.lower()][0]
        self.z_index = dataset_dir[name.lower()][1]
        self.series_num = dataset_dir[name.lower()][2]


        if traintest.lower() == "train":
            self.patient_list = sorted(glob.glob(self.dataset_path + "/imagesTr/*"))
        elif  traintest.lower() == "test":
            self.patient_list = sorted(glob.glob(self.dataset_path + "/imagesTs/*"))
        
        self.slice_list = []

        
        for patient in self.patient_list:
            for i in range(nib.load(patient).shape[dataset_dir[name.lower()][1]]):
                self.slice_list.append(patient + "__" + str(i))
        

    def __len__(self):

        return len(self.slice_list)

    def __getitem__(self, index):
        # ラベルが含まれるスライスのみを返すかどうか，要検討（やるならinitでも？）
        image_path, num = self.slice_list[index].split("__")
        num = int(num)
        label_path = image_path.replace("imagesTr", "labelsTr")

        # z軸の位置，軸の数，モダリティの数で場合分けが必要
        one_slice = nib.load(image_path).get_fdata()
        one_label = nib.load(label_path).get_fdata()
        if self.z_index == 1:
            one_slice = one_slice.transpose(1, 0, 2)
            one_slice = np.expand_dims(one_slice[num], 0)
            one_label = one_label.transpose(1, 0, 2)
            one_label = np.expand_dims(one_label[num], 0)
        
        elif self.series_num != 1:
            one_slice = one_slice.transpose(2, 3, 0, 1)
            one_slice = one_slice[num]
            one_label = one_label.transpose(2, 0, 1)
            one_label = np.expand_dims(one_label[num], 0)
        
        else:
            one_slice = one_slice.transpose(2, 0, 1)
            one_slice = np.expand_dims(one_slice[num], 0)
            one_label = one_label.transpose(2, 0, 1)
            one_label = np.expand_dims(one_label[num], 0)

        # たぶん正規化が必要

        return one_slice, one_label



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import pdb; pdb.set_trace()
    spleen = SegmentationDecathlon(name="spleen", traintest="train")
    spleen_loader = DataLoader(spleen, batch_size=5, shuffle=True)
    print(len(spleen))
    for i, data in enumerate(spleen_loader):
        print(data[0].shape)
        print(data[1].shape)