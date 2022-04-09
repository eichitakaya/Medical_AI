import glob
import torch
import numpy as np


class SegmentationDecathlon(torch.utils.data.Dataset):
    def __init__(self, name, traintest="train"):
        super().__init__()

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

        # dataフォルダのあるディレクトリの絶対パス(コマンドライン引数にしてもいいかも？)
        datafolder_path = "/takaya_workspace/data/decathlon"

        self.dataset_path = datafolder_path + dataset_dir[name.lower()]

        if traintest == "train":
            print(0)

    def __len__(self):
        return 
