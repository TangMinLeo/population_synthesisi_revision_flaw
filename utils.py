import os
import torch
# import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from investigation_function import investigate_functon
import numpy as np
import torch.nn as nn
import math


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        # 参数max_len表示一个句子的最大长度，也就是最多包含5000个词，d_model表示词向量的维度
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
        # return self.dropout(x)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


def get_data(args):
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize([72, 96]),  # args.image_size + 1/4 *args.image_size
    #     torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)

    out_samples, vocabulary = investigate_functon(args.file_path)
    # x = torch.LongTensor(out_samples)
    # out_emb = emb(x)
    # dataset = CustomDataset(out_emb)

    cate2base = torch.zeros(len(out_samples), len(out_samples[0]), len(vocabulary))
    for b in range(cate2base.size(0)):
        for i in range(cate2base.size(1)):
            idx = out_samples[b][i]
            cate2base[b, i, idx] = 1.0
            # small = 0.1/(args.voc_size-1)
            # for j in range(cate2base.size(2)):
            #     if j == idx:
            #         cate2base[b, i, j] = 0.9
            #     else:
            #         cate2base[b, i, j] = small

    cate2base = cate2base.cuda(torch.device("cuda:1"))

    print(cate2base[:2])
    dataset = CustomDataset(cate2base)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader, vocabulary


# def save_results(samples, path, **kwargs):
#     grid = torchvision.utils.make_grid(images, **kwargs)
#     ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)
