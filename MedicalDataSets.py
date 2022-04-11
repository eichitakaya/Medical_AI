import glob
import torch
import numpy as np
import nibabel as nib

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
        
        # dataフォルダのあるディレクトリの絶対パス(コマンドライン引数にしてもいいかも？)
        datafolder_path = "/takaya_workspace/data/decathlon/"

        self.dataset_path = datafolder_path + dataset_dir[name.lower()][0]


        if traintest.lower() == "train":
            self.patient_list = sorted(glob.glob(self.dataset_path + "/imagesTr/*"))
        elif  traintest.lower() == "test":
            self.patient_list = sorted(glob.glob(self.dataset_path + "/imagesTs/*"))
        
        self.slice_list = []

        
        for patient in self.patient_list:
            for i in range(nib.load(patient).shape[dataset_dir[name.lower()][1]]):
                self.slice_list.append(patient + "__" + str(i))
        

    def __len__(self):
        return 

    def __getitem__():
        return

if __name__ == "__main__":
    hippo = SegmentationDecathlon(name="heart", traintest="train")
    print(hippo.slice_list)