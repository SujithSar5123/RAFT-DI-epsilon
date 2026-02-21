# train.py
from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from core.raft import RAFT
import evaluate
import datasets
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except Exception:
    class GradScaler:
        def __init__(self, enabled=True): pass
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def step(self, optimizer): optimizer.step()
        def update(self): pass


MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear'
    )
    return optimizer, scheduler


class Logger:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        lr = self.scheduler.get_last_lr()[0]
        print("[{:6d}, {:10.7f}] ".format(self.total_steps, lr) + ("{:10.4f}, " * len(metrics_data)).format(*metrics_data))

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)

        self.running_loss = {}

    def push(self, metrics):
        self.total_steps += 1
        for k, v in metrics.items():
            self.running_loss[k] = self.running_loss.get(k, 0.0) + v

        if self.total_steps % SUM_FREQ == 0:
            self._print_training_status()

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
        for k, v in results.items():
            self.writer.add_scalar(k, v, self.total_steps)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def get_base_model(model):
    return model.module if hasattr(model, "module") else model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model = RAFT(args).to(device)
    print("Parameter Count: %d" % count_parameters(base_model))

    if device.type == 'cuda' and len(args.gpus) > 1:
        model = nn.DataParallel(base_model, device_ids=args.gpus)
    else:
        model = base_model

    if args.restore_ckpt is not None:
        ckpt = torch.load(args.restore_ckpt, map_location=device)
        get_base_model(model).load_state_dict(ckpt, strict=False)
        print(f"Restored checkpoint: {args.restore_ckpt}")

    model.train()

    if args.stage != 'chairs':
        get_base_model(model).freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(scheduler)

    os.makedirs("checkpoints", exist_ok=True)

    total_steps = 0
    while total_steps < args.num_steps:
        for data_blob in train_loader:
            optimizer.zero_grad(set_to_none=True)

            image1, image2, flow, valid = [x.to(device) for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 3.8)
                image1 = (image1 + stdv * torch.randn_like(image1)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn_like(image2)).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, gamma=args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)

            # Checkpoint + validation
            if (total_steps + 1) % VAL_FREQ == 0:
                ckpt_path = f"checkpoints/{total_steps+1}_{args.name}.pth"
                torch.save(get_base_model(model).state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                results = {}
                for val_dataset in (args.validation or []):
                    bm = get_base_model(model)
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(bm))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(bm))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(bm))

                if results:
                    logger.write_dict(results)

                model.train()
                if args.stage != 'chairs':
                    get_base_model(model).freeze_bn()

            total_steps += 1
            if total_steps >= args.num_steps:
                break

    logger.close()
    final_path = f"checkpoints/{args.name}.pth"
    torch.save(get_base_model(model).state_dict(), final_path)
    print(f"Saved final model: {final_path}")
    return final_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft_dic64', help="name your experiment")
    parser.add_argument('--stage', required=True, help="dataset stage (use dic64)")
    parser.add_argument('--restore_ckpt', default=None, help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use RAFT-small')
    parser.add_argument('--validation', type=str, nargs='+', default=[])

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--add_noise', action='store_true')

    parser.add_argument('--dataset_root', type=str, default=None,
                        help='root to dataset64 folder (contains train/ and test/)')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_augmentor', action='store_true',
                        help='use DIC64Augmentor (flip only)')

    args = parser.parse_args()

    if args.stage == 'dic64':
        args.image_size = [64, 64]
        if args.dataset_root is None:
            raise ValueError("--dataset_root is required for --stage dic64")

    torch.manual_seed(1234)
    np.random.seed(1234)

    train(args)

