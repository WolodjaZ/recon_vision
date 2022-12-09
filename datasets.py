import os
import csv
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


NORMS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "audiomnist": ([0.00382799, 0.00382799, 0.00382799], [0.06187078, 0.06187078, 0.05896849])
}

class ImagenetteDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                **kwargs):
        
        if split == "train":
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.RandomSizedCrop(min_max_height=(img_size, img_size), height=img_size, width=img_size, p=1.0),
                        A.HorizontalFlip(),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.2,0.2)),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
        else:
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
            
        data_dir = Path(data_path) / "imagenette2"
        csv_path = data_dir / "noisy_imagenette.csv"
        imgs = []
        with open(csv_path, "r") as f:
            fil = csv.reader(f)
            for img in fil:
                img_path = img[0]
                if split == "train":
                    if img_path.startswith("train"):
                        imgs.append(data_dir / img_path)
                else:
                    if img_path.startswith("val"):
                        imgs.append(data_dir / img_path)
        self.imgs = imgs
    
    def post_process(self, imgs):
        mean = torch.tensor(NORMS["imagenet"][0]).view(-1, 1, 1)
        std = torch.tensor(NORMS["imagenet"][1]).view(-1,1,1)
        return imgs * std + mean
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image_filepath = self.imgs[idx]
        image = cv2.imread(str(image_filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return image


class AudioMnistDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                **kwargs):
            
        data_dir = Path(data_path) / "AudioMNIST"
        if split == "train":
            data_dir = data_dir / "train"
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.RandomSizedCrop(min_max_height=(img_size, img_size), height=img_size, width=img_size, p=1.0),
                        A.HorizontalFlip(),
                        A.Normalize(mean=NORMS["audiomnist"][0], std=NORMS["audiomnist"][1]),
                        ToTensorV2(),
                    ]
            )
        else:
            data_dir = data_dir / "test"
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
            
        imgs = []
        for filename in os.listdir(data_dir):
            path = data_dir / filename
            imgs.append(path)
        self.imgs = imgs
    
    def post_process(self, imgs):
        mean = torch.tensor(NORMS["audiomnist"][0]).view(-1, 1, 1)
        std = torch.tensor(NORMS["audiomnist"][1]).view(-1,1,1)
        return imgs * std + mean
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image_filepath = self.imgs[idx]
        image = cv2.imread(str(image_filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[59:427,81:576]
        
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return image


class AIDDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                **kwargs):
        
        data_dir = Path(data_path) / "AID"
        if split == "train":
            data_dir = data_dir / "train"
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.RandomSizedCrop(min_max_height=(img_size, img_size), height=img_size, width=img_size, p=1.0),
                        A.HorizontalFlip(),
                        A.Normalize(mean=NORMS["audiomnist"][0], std=NORMS["audiomnist"][1]),
                        ToTensorV2(),
                    ]
            )
        else:
            data_dir = data_dir / "test"
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
            
        imgs = []
        for filename in os.listdir(data_dir):
            path = data_dir / filename
            imgs.append(path)
        self.imgs = imgs
    
    def post_process(self, imgs):
        mean = torch.tensor(NORMS["imagenet"][0]).view(-1, 1, 1)
        std = torch.tensor(NORMS["imagenet"][1]).view(-1,1,1)
        return imgs * std + mean
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        image_filepath = self.imgs[idx]
        image = cv2.imread(str(image_filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        
        return image


class OXFORDIIITPETDataset(torchvision.datasets.OxfordIIITPet):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                 download: bool = False,
                **kwargs):
        
        if split == "train":
            split_data = "trainval"
        else:
            split_data = "test"
        
        super().__init__(
            root=data_path,
            split=split_data,
            target_types="segmentation",
            download=download
        )
        
        if split == "train":
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.HorizontalFlip(),
                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.2,0.2)),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
        else:
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
    
    def _preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask
        
    def __getitem__(self, idx: int):
        image_filepath = self._images[idx]
        image = cv2.imread(str(image_filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target = cv2.imread(str(self._segs[idx]), cv2.IMREAD_UNCHANGED)
        
        target = self._preprocess_mask(target)
        if self.transforms:
            augmented = self.transforms(image=image, mask=target)
            image, target = augmented['image'], augmented['mask']

        return image, target     


class VOCSegmentationDataset(torchvision.datasets.VOCSegmentation):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                 download: bool = False,
                **kwargs):
        
        if split == "train":
            split_data = "train"
        else:
            split_data = "val"
        
        super().__init__(
            root=data_path,
            image_set=split_data,
            download=download
        )
        
        if split == "train":
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.HorizontalFlip(),
                        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=25, p=0.5),
                        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.2,0.2)),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
        else:
            self.transforms = A.Compose(
                    [
                        A.Resize(height=img_size, width=img_size),
                        A.Normalize(mean=NORMS["imagenet"][0], std=NORMS["imagenet"][1]),
                        ToTensorV2(),
                    ]
            )
    
    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(VOC_COLORMAP)), dtype=np.float32)
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        return segmentation_mask
    
    def __getitem__(self, idx: int):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)
        
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        return image, mask  