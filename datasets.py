# datasets.py
# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import os
import random
from glob import glob
import os.path as osp

import numpy as np
import torch
import torch.utils.data as data

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

try:
    from utils.augmentor import DIC64Augmentor
except Exception:
    DIC64Augmentor = None


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, enforce_size=None):
        self.augmentor = None
        self.sparse = sparse
        self.enforce_size = enforce_size  # (H,W) or None

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def _maybe_init_seed(self):
        if self.init_seed:
            return
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            torch.manual_seed(worker_info.id)
            np.random.seed(worker_info.id)
            random.seed(worker_info.id)
        self.init_seed = True

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        self._maybe_init_seed()

        index = index % len(self.image_list)

        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # Flow must be HxWx2
        if flow.ndim != 3 or flow.shape[2] != 2:
            raise ValueError(f"Flow must be HxWx2, got {flow.shape} from {self.flow_list[index]}")

        # Force grayscale -> 3ch (RAFT expects 3ch)
        if img1.ndim == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]

        if img2.ndim == 2:
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img2 = img2[..., :3]

        # Augmentation (if any)
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # Enforce fixed size (critical for dic64)
        if self.enforce_size is not None:
            H, W = self.enforce_size
            if img1.shape[:2] != (H, W):
                raise ValueError(f"img1 wrong size {img1.shape[:2]} expected {(H, W)}")
            if img2.shape[:2] != (H, W):
                raise ValueError(f"img2 wrong size {img2.shape[:2]} expected {(H, W)}")
            if flow.shape[:2] != (H, W):
                raise ValueError(f"flow wrong size {flow.shape[:2]} expected {(H, W)}")

        # To torch CHW
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid).float()
        else:
            valid = ((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float()

        return img1, img2, flow, valid

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


class RAFTDIC64(FlowDataset):
    """
    root/train/ref/pair_00001.png
    root/train/def/pair_00001.png
    root/train/flow/pair_00001.flo
    and similarly for root/test/...
    """
    def __init__(self, root, split="train", enforce_size=(64, 64)):
        super().__init__(aug_params=None, sparse=False, enforce_size=enforce_size)

        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        ref_dir  = osp.join(root, split, "ref")
        def_dir  = osp.join(root, split, "def")
        flow_dir = osp.join(root, split, "flow")

        ref_list  = sorted(glob(osp.join(ref_dir,  "pair_*.png")))
        def_list  = sorted(glob(osp.join(def_dir,  "pair_*.png")))
        flow_list = sorted(glob(osp.join(flow_dir, "pair_*.flo")))

        if len(ref_list) == 0:
            raise FileNotFoundError(f"No ref images found in {ref_dir}")
        if not (len(ref_list) == len(def_list) == len(flow_list)):
            raise ValueError(f"Count mismatch in {split}: ref={len(ref_list)}, def={len(def_list)}, flow={len(flow_list)}")

        self.image_list = [[r, d] for r, d in zip(ref_list, def_list)]
        self.flow_list = flow_list
        self.is_test = False


# --- Original RAFT datasets (unchanged) ---

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super().__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]
                self.extra_info += [(scene, i)]

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super().__init__(aug_params)
        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split == 'training' and xid == 1) or (split == 'validation' and xid == 2):
                self.flow_list += [flows[i]]
                self.image_list += [[images[2*i], images[2*i+1]]]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super().__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [[images[i], images[i+1]]]
                            self.flow_list += [flows[i]]
                        else:
                            self.image_list += [[images[i+1], images[i]]]
                            self.flow_list += [flows[i+1]]


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super().__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [[frame_id]]
            self.image_list += [[img1, img2]]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super().__init__(aug_params, sparse=True)

        seq_ix = 0
        while True:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))
            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [[images[i], images[i+1]]]
            seq_ix += 1


def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """Create the dataloader for training."""

    if args.stage == 'dic64':
        if args.dataset_root is None:
            raise ValueError("--dataset_root is required for stage='dic64'")

        train_dataset = RAFTDIC64(
            root=args.dataset_root,
            split='train',
            enforce_size=tuple(args.image_size)
        )

        # Optional: flip-only augmentor (safe for DIC)
        if getattr(args, "use_augmentor", False) and DIC64Augmentor is not None:
            train_dataset.augmentor = DIC64Augmentor(do_flip=True)

        pin = torch.cuda.is_available()
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=pin,
            shuffle=True,
            num_workers=getattr(args, "num_workers", 2),
            drop_last=True
        )

        print(f"Training with {len(train_dataset)} image pairs")
        return train_loader

    # Legacy RAFT training stages
    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')

    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        train_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass') + FlyingThings3D(aug_params, dstype='frames_finalpass')

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things
        else:
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    pin = torch.cuda.is_available()
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=pin,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 4),
        drop_last=True
    )

    print(f"Training with {len(train_dataset)} image pairs")
    return train_loader
