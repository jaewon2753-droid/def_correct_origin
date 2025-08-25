import torch
import torch.nn as nn
import torch.nn.functional as F
from .basicBlocks import DoubleConv, Down, Up, OutConv

# --------------------------------------------------------------------------- #
# 트랜스포머 블록 정의 (U-Net의 병목 구간에 삽입)
# --------------------------------------------------------------------------- #

class TransformerBlock(nn.Module):
    def __init__(self, channels, n_heads, n_layers, patch_size=2, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Conv2d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=0.1, activation='relu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.reshape_conv = nn.Conv2d(embed_dim, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_res = x
        
        x = self.patch_embedding(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2).view(b, -1, h // self.patch_size, w // self.patch_size)
        
        # ========================================================== #
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 수정된 부분 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
        # ---------------------------------------------------------- #
        # 작아진 크기(h/2, w/2)를 원래 크기(h, w)로 되돌리는 업샘플링 코드를 추가합니다.
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        # ========================================================== #
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
        # ========================================================== #
        
        x = self.reshape_conv(x)
        
        return x + x_res # 이제 크기가 같아져서 덧셈이 가능합니다.

# --------------------------------------------------------------------------- #
# U-Net과 트랜스포머를 결합한 최종 생성자 모델
# --------------------------------------------------------------------------- #

class UNetTransformer(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetTransformer, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        
        self.transformer = TransformerBlock(channels=256, n_heads=8, n_layers=4, patch_size=2)
                                             
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        input_image = x
        
        x1 = self.inc(input_image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x_transformer = self.transformer(x3)
        
        x = self.up1(x_transformer, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        
        return torch.tanh(logits + input_image)
