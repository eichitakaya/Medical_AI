import torch
import torch.nn as nn

class VitInputLayer(nn.module):
    def __init__(self,
        in_channels:int = 3, 
        emb_dim:int = 384,
        num_patch_row:int=2,
        image_size:int=32
        ):

        super(VitInputLayer, self).__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.numpatch_row**2

        self.patch_size = int(self.image_size // self.num_patch_row)

        self.patch_emb_layer = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.emb_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, emb_dim)
        )

        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patch+1, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        

        