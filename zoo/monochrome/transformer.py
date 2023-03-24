import math

import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CNNHead(nn.Module):
    def __init__(self, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            #nn.Conv1d(in_chans, embed_dim // 2, kernel_size=7, stride=2),
            #nn.BatchNorm1d(embed_dim // 2),
            #nn.SiLU(),
            #nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, stride=2),
            nn.Conv1d(in_chans, embed_dim, kernel_size=1, stride=1),
            Rearrange('b h n -> n b h'),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):  # x:[B,ch,N_seq]
        x = self.proj(x)
        return x


class SigTransformer(nn.Module):
    __model_name__ = 'transformer'

    def __init__(self, in_ch=3, n_cls=2, hidden=512, nlayers=16, dropout=0.1, seq_len=180):
        super(SigTransformer, self).__init__()
        nhead = hidden // 64

        self.head = CNNHead(in_ch, hidden)
        # self.pos_encoder = PositionalEncoding(hidden, dropout)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len + 1, 1, hidden) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(hidden, nhead, dim_feedforward=2048, dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers, encoder_norm)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_cls)
        )

    def forward(self, src):
        src = self.head(src)  # [N,B,h]
        cls_tokens = repeat(self.cls_token, '1 1 h -> 1 b h', b=src.shape[1])
        src = torch.cat((cls_tokens, src), dim=0)
        # src = self.pos_encoder(src)
        src += self.pos_embedding
        src = self.pos_drop(src)

        output = rearrange(self.encoder(src), 'n b h -> b n h')
        output = self.mlp_head(output[:, 0, :])

        return output


if __name__ == '__main__':
    from thop import profile

    transformer = SigTransformer()
    x = torch.randn(1, 3, 180)

    flops, params = profile(transformer, (x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
