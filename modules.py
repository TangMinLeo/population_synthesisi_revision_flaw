import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PositionalEncoding
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # print(f"x is on cuda: {x.is_cuda}")
        # print(f"t is on cuda: {t.is_cuda}")
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=512):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class SelfAttention(nn.Module):
    # d_model表示离散值嵌入为向量时向量的长度
    # size表示一条样本对应着多少个嵌入向量
    def __init__(self, d_model, size):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.size = size
        self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.ln = nn.LayerNorm([d_model])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([d_model]),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        # x = x.view(-1, self.channels, self.size[0] * self.size[1]).swapaxes(1, 2)
        assert x.size(1) == self.size, '输入的张量中词的个数与模型中的不一致'
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        # return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size[0], self.size[1])
        return attention_value


class Net_model(nn.Module):
    def __init__(self, c_in=13, c_out=13, time_dim=512, voca_size=70, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.voca_size = voca_size

        self.in_liner = nn.Linear(voca_size, time_dim)
        self.pos_enc = PositionalEncoding(time_dim, 0.0)
        # self.pos_enc = PositionalEncoding(voca_size, 0.0)

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(256, 128)

        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(128, 256)

        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(64, 256)
        # self.sa3 = SelfAttention(256, [9, 12])

        self.bot1 = DoubleConv(256, 512)
        self.botsa1 = SelfAttention(64, 512)
        self.bot2 = DoubleConv(512, 512)
        self.botsa2 = SelfAttention(64, 512)
        self.bot3 = DoubleConv(512, 256)
        self.botsa3 = SelfAttention(64, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 128)
        # self.sa4 = SelfAttention(128, [18, 24])
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(256, 64)
        # self.sa5 = SelfAttention(64, [36, 48])
        self.up3 = Up(128, 64)

        self.sa6 = SelfAttention(512, 64)
        # self.sa6 = SelfAttention(64, [72, 96])

        self.outc = nn.Conv1d(64, c_out, kernel_size=1)

        self.out_liner = nn.Linear(time_dim, voca_size)


    def pos_encoding(self, t, channels):
        # pos_enc = torch.zeros(len(t), channels)
        # for pos in t:
        #     for i in range(channels//2):
        #         pos_enc[pos, 2*i] = math.sin(pos/(10000**((2*i)/channels)))
        #         pos_enc[pos, 2*i+1] = math.cos(pos/(10000**((2*i)/channels)))
        # return pos_enc

        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        # print(f'inv_freq  is on cuda: {inv_freq.is_cuda}')
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        pos_enc = torch.zeros(pos_enc_a.size(0), 2*pos_enc_a.size(1)).to(self.device)
        pos_enc[:, 0::2] = pos_enc_a
        pos_enc[:, 1::2] = pos_enc_b
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # print(f't in model forward is on cuda: {t.is_cuda}')
        t = t.unsqueeze(-1).type(torch.float)
        # 对时刻的编码0:T-1共T个时刻
        t = self.pos_encoding(t, self.time_dim)
        t.to(self.device)

        # 线性变换之后再加位置编码
        x0 = self.in_liner(x)
        x0 = self.pos_enc(x0)

        # 线性变换之前（one-hot编码基础上直接加）加位置编码
        # x0 = self.pos_enc(x)
        # x0 = self.in_liner(x0)

        x1 = self.inc(x0)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.botsa1(x4)
        x4 = self.bot2(x4)
        x4 = self.botsa2(x4)
        x4 = self.bot3(x4)
        x4 = self.botsa3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)

        dis_out = self.out_liner(output)

        return dis_out
