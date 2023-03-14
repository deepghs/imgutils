import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
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
            nn.Conv1d(in_chans, embed_dim // 2, kernel_size=7, stride=2),
            nn.BatchNorm1d(embed_dim // 2),
            nn.SiLU(),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=5, stride=2),
        )

    def forward(self, x):  # x:[B,ch,N_seq]
        x = self.proj(x).permute(2, 0, 1)
        return x


class SigTransformer(nn.Module):
    __model_name__ = 'transformer'

    def __init__(self, in_ch=3, n_cls=2, n_query=8, hidden=384, nlayers=3, dropout=0.1):
        super(SigTransformer, self).__init__()
        nhead = hidden // 64

        self.encoder = CNNHead(in_ch, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Embedding(n_query, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.transformer = nn.Transformer(
            d_model=hidden, nhead=nhead, num_encoder_layers=nlayers,
            num_decoder_layers=nlayers, dim_feedforward=hidden, dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden, n_cls)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src):
        trg = self.decoder.weight

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        # src_pad_mask = self.make_len_mask(src)
        # trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = trg.unsqueeze(1).repeat(1, src.shape[1], 1)
        # trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
        output = self.transformer(src, trg, tgt_mask=self.trg_mask).transpose(0, 1)  # [B,N,emb]
        output = self.fc_out(output[:, 0, :])

        return output


if __name__ == '__main__':
    transformer = SigTransformer()
    x = torch.randn(8, 3, 380)
    y = transformer(x)
    print(y.shape)
