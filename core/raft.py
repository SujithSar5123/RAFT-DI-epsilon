# raft.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.update import BasicUpdateBlock, SmallUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from corr.corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import coords_grid

try:
    autocast = torch.cuda.amp.autocast
except Exception:
    class autocast:
        def __init__(self, enabled): self.enabled = enabled
        def __enter__(self): return None
        def __exit__(self, *args): return False


class RAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Force your desired corr settings
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.args.corr_levels = 2
            self.args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 2
            self.args.corr_radius = 4

        if not hasattr(self.args, "dropout"):
            self.args.dropout = 0.0
        if not hasattr(self.args, "alternate_corr"):
            self.args.alternate_corr = False
        if not hasattr(self.args, "mixed_precision"):
            self.args.mixed_precision = False

        # feature, context, update
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=self.args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=self.args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)
        return coords0, coords1

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        # normalize to [-1,1]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim, cdim = self.hidden_dim, self.context_dim

        # features
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # context
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                net, _, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            # Full-resolution prediction (your DIC requirement)
            flow_predictions.append(coords1 - coords0)

        if test_mode:
            return coords1 - coords0, flow_predictions[-1]

        return flow_predictions
