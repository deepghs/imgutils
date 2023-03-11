import lpips
import torch
from torch import nn


class LPIPSFeature(nn.Module):
    def __init__(self, model: lpips.LPIPS):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, in0):
        in0 = in0 * 2 - 1
        in0_input = self.model.scaling_layer(in0) if self.model.version == '0.1' else in0
        outs0 = self.model.net.forward(in0_input)
        feats0 = []
        for kk in range(self.model.L):
            feats0.append(lpips.normalize_tensor(outs0[kk]))
        return tuple(feats0)


class LPIPSDiff(nn.Module):
    def __init__(self, model: lpips.LPIPS):
        nn.Module.__init__(self)
        self.model = model

    def forward(self, *all_feats):
        n = len(all_feats) // 2
        feats0, feats1 = all_feats[:n], all_feats[n:]
        diffs = {}
        for kk in range(self.model.L):
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.model.lpips:
            res = [lpips.spatial_average(self.model.lins[kk](diffs[kk]), keepdim=True)
                   for kk in range(self.model.L)]
        else:
            res = [lpips.spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True)
                   for kk in range(self.model.L)]

        return torch.stack(res).sum(axis=0).reshape(-1)
