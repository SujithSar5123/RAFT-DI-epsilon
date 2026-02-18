# augmentor.py
import numpy as np
from PIL import Image
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

from torchvision.transforms import ColorJitter


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale

        # You set these to 0.0 (good for DIC-style "no resize/crop" if you ever use it)
        self.spatial_aug_prob = 0.0
        self.stretch_prob = 0.0
        self.max_stretch = 0.2

        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # Contrast/brightness only (good for speckle; avoid hue/saturation)
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
        else:
            stack = np.concatenate([img1, img2], axis=0)
            stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
            img1, img2 = np.split(stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=(50, 100)):
        """Occlusion augmentation (supports grayscale or RGB)."""
        ht, wd = img1.shape[:2]
        if np.random.rand() >= self.eraser_aug_prob:
            return img1, img2

        if img2.ndim == 2:
            mean_val = float(np.mean(img2))
        else:
            mean_val = np.mean(img2.reshape(-1, 3), axis=0)

        for _ in range(np.random.randint(1, 3)):
            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])

            if img2.ndim == 2:
                img2[y0:y0+dy, x0:x0+dx] = mean_val
            else:
                img2[y0:y0+dy, x0:x0+dx, :] = mean_val

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        ht, wd = img1.shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
            if np.random.rand() < self.v_flip_prob:
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # crop (only if crop_size fits)
        if img1.shape[0] < self.crop_size[0] or img1.shape[1] < self.crop_size[1]:
            return img1, img2, flow

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + 1)
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1] + 1)

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        return np.ascontiguousarray(img1), np.ascontiguousarray(img2), np.ascontiguousarray(flow)


class SparseFlowAugmentor:
    # leaving as-is from RAFT conventions (not used for dic64)
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        self.do_flip = do_flip
        self.h_flip_prob = 0.5

        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3)
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        stack = np.concatenate([img1, img2], axis=0)
        stack = np.array(self.photo_aug(Image.fromarray(stack)), dtype=np.uint8)
        img1, img2 = np.split(stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2, bounds=(50, 100)):
        ht, wd = img1.shape[:2]
        if np.random.rand() >= self.eraser_aug_prob:
            return img1, img2

        if img2.ndim == 2:
            mean_val = float(np.mean(img2))
        else:
            mean_val = np.mean(img2.reshape(-1, 3), axis=0)

        for _ in range(np.random.randint(1, 3)):
            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(bounds[0], bounds[1])
            dy = np.random.randint(bounds[0], bounds[1])

            if img2.ndim == 2:
                img2[y0:y0+dy, x0:x0+dx] = mean_val
            else:
                img2[y0:y0+dy, x0:x0+dx, :] = mean_val

        return img1, img2

    # NOTE: resize_sparse_flow_map + spatial_transform omitted here because not relevant to your dic64 stage


class DIC64Augmentor:
    """Safe augmentation for 64x64 DIC: flips only. No crop/resize/photometric."""
    def __init__(self, do_flip=True):
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

    def __call__(self, img1, img2, flow):
        if not self.do_flip:
            return img1, img2, flow

        if np.random.rand() < self.h_flip_prob:
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            flow = flow[:, ::-1] * [-1.0, 1.0]

        if np.random.rand() < self.v_flip_prob:
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            flow = flow[::-1, :] * [1.0, -1.0]

        return np.ascontiguousarray(img1), np.ascontiguousarray(img2), np.ascontiguousarray(flow)
