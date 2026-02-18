# corr.py
import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler

try:
    import alt_cuda_corr
    HAS_ALT = True
except Exception:
    HAS_ALT = False


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)  # B,H,W,2
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), dim=-1)  # (2r+1,2r+1,2)

            centroid = coords.reshape(batch*h1*w1, 1, 1, 2) / (2**i)
            coords_lvl = centroid + delta.view(1, 2*r+1, 2*r+1, 2)

            corr_sampled = bilinear_sampler(corr, coords_lvl)  # (B*H*W, dim, 2r+1, 2r+1)
            corr_sampled = corr_sampled.view(batch, h1, w1, -1)
            out_pyramid.append(corr_sampled)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        denom = fmap1.new_tensor(dim).float().sqrt()
        return corr / denom


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=2, radius=4):
        if not HAS_ALT:
            raise RuntimeError("alt_cuda_corr not available. Compile it or set args.alternate_corr=False.")
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for _ in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)  # B,H,W,2
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_lvl, fmap2_lvl = self.pyramid[i]

            fmap1_i = fmap1_lvl.permute(0, 2, 3, 1).contiguous()
            fmap2_i = fmap2_lvl.permute(0, 2, 3, 1).contiguous()

            coords_lvl = coords / (2**i)
            coords_i = coords_lvl.reshape(B, 1, H, W, 2).contiguous()

            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1).reshape(B, -1, H, W)
        denom = corr.new_tensor(dim).float().sqrt()
        return corr / denom
