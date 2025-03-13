import os
import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

import pydicom

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as L
from pytorch_lightning import seed_everything


class VinBigDataCXR(Dataset):
    def __init__(self, img_dir, annotations, transform = None):
        super().__init__()
        self.img_dir = img_dir
        self.annotations = annotations
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_id = row['image_id']
        class_name = row['class_name']
        label = row['class_id']
        rad_id = row['rad_id']
        x_min, y_min, x_max, y_max = row[['x_min', 'y_min', 'x_max', 'y_max']]

        dicom_path = os.path.join(self.img_dir, f"{img_id}.dicom")
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        image = Image.fromarray(image.squeeze(), mode='L')

        width, height = image.size
        bbox = torch.tensor([x_min/width, y_min/height, (x_max-x_min)/width, (y_max-y_min)/height])

        if self.transform:
            image = self.transform(image)
        
        return image, label, bbox

class VinBigDataCXRDatamodule(L.LightningDataModule):
    def __init__(self, img_dir, csv_file, batch_size=32, num_workers=16, val_split=0.2):
        super().__init__()
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.train_transform = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,), std=(0.229,))
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485,), std=(0.229,))
        ])
    
    def setup(self, stage=None):
        annotations = pd.read_csv(self.csv_file)

        train_annotations, val_annotations = train_test_split(
            annotations, test_size=self.val_split, random_state=42
        )
        
        self.train_dataset = VinBigDataCXR(img_dir=self.img_dir, annotations=train_annotations, transform=self.train_transform)
        self.val_dataset = VinBigDataCXR(img_dir=self.img_dir, annotations=val_annotations, transform=self.val_transform)

        if stage == "fit":
            print(f"Training Set Size: {len(self.train_dataset)}")
            print(f"Validation Set Size: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# img_dir = 'VinBigDataCXR/train' 
# csv_file = 'VinBigDataCXR/train.csv'

# data_module = VinBigDataCXRDatamodule(img_dir=img_dir, csv_file=csv_file, batch_size=16, num_workers=4)

# data_module.setup(stage="fit")

# train_dataloader = data_module.train_dataloader()
# val_dataloader = data_module.val_dataloader()

# for batch_idx, (images, labels, bboxes) in enumerate(train_dataloader):
#     print(f"Batch {batch_idx + 1} from Train:")
#     print(f"Images shape: {images.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Bboxes shape: {bboxes.shape}")
#     break

# for batch_idx, (images, labels, bboxes) in enumerate(val_dataloader):
#     print(f"Batch {batch_idx + 1} from Val:")
#     print(f"Images shape: {images.shape}")
#     print(f"Labels shape: {labels.shape}")
#     print(f"Bboxes shape: {bboxes.shape}")
#     break