import glob
from importlib.resources import path
import torch
import numpy as np


class SegmentationDecathlon(torch.utils.data.Dataset):
    def __init__(self, name, traintest="train"):
        super().__init__()
        """

        """
        assert traintest.lower() == "train" or traintest.lower() == "test", "traintest must be 'train' or 'test'."

        self.dataset_dir = {
            "brain": "Task01_BrainTumour",
            "heart": "Task02_Heart",
            "liver": "Task03_Liver",
            "hippocampus": "Task04_Hippocampus",
            "prostate": "Task05_Prostate",
            "lung": "Task06_Lung",
            "pancreas": "Task07_Pancreas",
            "hepaticvessel": "Task08_HepaticVessel",
            "spleen": "Task09_Spleen",
            "colon": "Task10_Colon"
        }
        assert name.lower() in self.dataset_dir, "The Spefified dataset {} does not exist.".format(name)
        
        # dataフォルダのあるディレクトリの絶対パス(コマンドライン引数にしてもいいかも？)
        datafolder_path = "/takaya_workspace/data/decathlon/"

        self.dataset_path = datafolder_path + self.dataset_dir[name.lower()]


        if traintest.lower() == "train":
            self.patient_list = glob.glob(self.dataset_path + "/imagesTr/*")
        elif  traintest.lower() == "test":
            self.patient_list = glob.glob(self.dataset_path + "/imagesTs/*")

    def __len__(self):
        return 

    def __getitem__():
        return

if __name__ == "__main__":
    hippo = SegmentationDecathlon(name="HippoCampus", traintest="train")
    print(hippo.dataset_dir)
    print(hippo.dataset_path)
    print(hippo.patient_list)