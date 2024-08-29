import torch
import torch.nn as nn
import torch.nn.functional as F

class VitInputLayer(nn.Module):
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

        self.num_patch = self.num_patch_row**2

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
        """
        x: (B, C, H, W)
        z_0: (B, N, D)
        """

        z_0 = self.patch_emb_layer(x) # (B, D, H/P, W/P)
        z_0 = z_0.flatten(2) # (B, D, Np)
        z_0 = z_0.transpose(1, 2) # (B, Np, D)

        z_0 = torch.cat([self.cls_token.repeat(repeats=(x.size(0),1,1)), z_0], dim=1) # (B, (Np+1), D)

        z_0 = z_0 + self.pos_emb # (B, (Np+1), D)

        return z_0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
        emb_dim:int=384,
        head:int=3,
        dropout:float=0.
    ):

        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // head
        self.sqrt_dh = self.head_dim**0.5

        self.w_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.w_v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, D)
        out: (B, N, D)
        """

        batch_size, num_patch, _ = z.size()
        
        # (B, N, D) -> (B, N, D)
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)

        # (B, N, D) -> (B, N, h, D//h)
        q = q.view(batch_size, num_patch, self.head, self.head_dim)
        k = k.view(batch_size, num_patch, self.head, self.head_dim)
        v = v.view(batch_size, num_patch, self.head, self.head_dim)

        # (B, N, h, D//h) -> (B, h, N, D//h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (B, h, N, D//h) -> (B, h, D//h, N)
        k_T = k.transpose(2, 3)

        # (B, h, N, D//h) × (B, h, D//h, N) -> (B, h, N, N)
        dots = (q @ k_T) / self.sqrt_dh

        attn = F.softmax(dots, dim=-1)
        attn = self.attn_drop(attn)

        # (B, h, N, N) × (B, h, N, D//h) -> (B, h, N, D//h)
        out = attn @ v
        # (B, h, N, D//h) -> (B, N, h, D//h)
        out = out.transpose(1, 2)
        # (B, N, h, D//h) -> (B, N, D)
        out = out.reshape(batch_size, num_patch, self.emb_dim)

        # (B, N, D) -> (B, N, D)
        out = self.w_o(out)

        return out

class VitEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_dim:int=384,
        head:int=8,
        hidden_dim:int=384*4,
        dropout:float=0.
        ):

        super(VitEncoderBlock, self).__init__()

        self.ln1 = nn.LayerNorm(emb_dim)

        self.msa = MultiHeadSelfAttention(
            emb_dim=emb_dim,
            head=head,
            dropout = dropout,
        )

        self.ln2 = nn.LayerNorm(emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, D)
        out: (B, N, D)
        """

        out = self.msa(self.ln1(z)) + z
        out = self.mlp(self.ln2(out)) + out
        
        return out

class Vit(nn.Module):
    def __init__(self,
        in_channels:int=3,
        num_classes:int=10,
        emb_dim:int=384,
        num_patch_row:int=2,
        image_size:int=32,
        num_blocks:int=7,
        head:int=8,
        hidden_dim:int=384*4,
        dropout:float=0.
        ):

        super(Vit, self).__init__()

        self.input_layer = VitInputLayer(
            in_channels,
            emb_dim,
            num_patch_row,
            image_size)
        
        self.encoder = nn.Sequential(*[
            VitEncoderBlock(
                emb_dim=emb_dim,
                head=head,
                hidden_dim=hidden_dim,
                dropout = dropout
            )
            for _ in range(num_blocks)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        out: (B, M)
        """

        # (B, C, H, W) -> (B, N, D)
        out = self.input_layer(x)

        # (B, N, D) -> (B, N, D)
        out = self.encoder(out)

        # (B, N, D) -> (B, D)
        cls_token = out[:,0]

        # (B, D) -> (B, M)
        pred = self.mlp_head(cls_token)
        
        return pred

if __name__ == '__main__':
    num_classes = 10
    batch_size, channel, height, width = 2, 3, 32, 32
    x = torch.randn(batch_size, channel, height, width)
    
    vit = Vit(in_channels=channel, num_classes=num_classes)
    pred = vit(x)

    print(pred.shape)

