import csv
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


NORMS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

class ImagenetteDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 img_size: int,
                **kwargs):
        
        self.transforms = A.Compose(
                [
                    A.HorizontalFlip(),
                    A.Resize(height=img_size, width=img_size),
                    A.RandomSizedCrop(min_max_height=(img_size, img_size), height=img_size, width=img_size, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.2,0.2)),
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
