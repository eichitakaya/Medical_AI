import glob
import torch
import numpy as np
import nibabel as nib
from PIL import Image

class MedMNIST(torch.utils.data.Dataset):
    def __init__(self, name, traintestval="train"):
        super().__init__()
        """
        ここで全てロードしておき，getitemではそこから直接取り出す
        """
        assert traintestval.lower() == "train" or traintestval.lower() == "val" or traintestval.lower() == "test", "traintest must be 'train' or 'val' or 'test'."

        # データセット名を保持
        dataset_dir = {
            "adrenal3d": ("adrenalmnist3d.npz", 28, 1, 2),
            "fracture3d": ("fracturemnist3d.npz", 28, 1, 3),
            "organ3d": ("organmnist3d.npz", 28, 1, 11),
            "synapse3d": ("synapsemnist3d.npz", 28, 1, 2),
            "blood": ("bloodmnist.npz", 1, 3, 8),
            "nodule3d": ("nodulemnist3d.npz", 28, 1, 2),
            "organs": ("organsmnist.npz", 1, 1, 11),
            "tissue": ("tissuemnist.npz", 1, 1, 8),
            "breast": ("breastmnist.npz", 1, 1, 2),
            "oct": ("octmnist.npz", 1, 1, 4),
            "path": ("pathmnist.npz", 1, 3, 9),
            "vessel3d": ("vesselmnist3d.npz", 28, 1, 2),
            "chest": ("chestmnist.npz", 1, 1, 2),
            "organa": ("organamnist.npz", 1, 1, 11),
            "pneumonia": ("pneumoniamnist.npz", 1, 1, 2),
            "derma": ("dermamnist.npz", 1, 3, 7),
            "organc": ("organcmnist.npz", 1, 1, 11),
            "retina": ("retinamnist.npz", 1, 3, 5)
        }
        assert name.lower() in dataset_dir, "The Spefified dataset {} does not exist.".format(name)
        
        datafolder_path = "/takaya_workspace/Medical_AI/data/medmnist/"

        self.dataset_path = datafolder_path + dataset_dir[name.lower()][0]
        self.n_slices = dataset_dir[name.lower()][1]
        self.n_channels = dataset_dir[name.lower()][2]
        self.n_class = dataset_dir[name.lower()][3]
        
        self.name = name.lower()
                
        npz = np.load(self.dataset_path)
        

        if traintestval.lower() == "train":
            self.data = torch.tensor(npz["train_images"], dtype=torch.float64)
            self.targets = torch.tensor(npz["train_labels"], dtype=torch.int64)
        elif  traintestval.lower() == "val":
            self.data = torch.tensor(npz["val_images"], dtype=torch.float64)
            self.targets = torch.tensor(npz["val_labels"], dtype=torch.int64)
        elif traintestval.lower() == "test":
            self.data = torch.tensor(npz["test_images"], dtype=torch.float64)
            self.targets = torch.tensor(npz["test_labels"], dtype=torch.int64)

    def __len__(self):

        return len(self.targets)

    def __getitem__(self, index):
        image = self.data[index].float()

        # カラーだったらtranspose
        if image.dim() > 2:
            image = torch.permute(image, (2,0,1))
            
        label = self.targets[index]
        
        # chestだったらラベルに前処理が必要
        if self.name == "chest":
            if label.sum() > 0:
                label = torch.tensor(1)
            else:
                label = torch.tensor(0)
            return image, label
        else:
            return image, label[0]



if __name__ == "__main__":
    from MedMNIST import MedMNIST

    train_dataset = MedMNIST("adrenal3d", traintestval="train")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 100,
        shuffle = True
    )
    
    print(train_dataset.targets)