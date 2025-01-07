import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class LungCancerDataset(Dataset):
    def __init__(self, image_dir, transform=None, metric_learning=False):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.metric_learning = metric_learning

        # Map folder names to labels
        self.class_map = {
            folder_name: idx
            for idx, folder_name in enumerate(os.listdir(image_dir))
            if os.path.isdir(os.path.join(image_dir, folder_name))
        }

        self.image_paths = []
        self.labels = []
        self.class_to_images = {lbl: [] for lbl in self.class_map.values()}

        for class_name, label in self.class_map.items():
            class_dir = os.path.join(image_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, file_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.class_to_images[label].append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        if not self.metric_learning:
            # standard classification
            return image, label
        else:
            # triplet for metric learning
            anchor_image = image
            anchor_label = label

            # Positive: same class
            positive_image_path = random.choice(self.class_to_images[anchor_label])
            positive_image = Image.open(positive_image_path).convert("RGB")
            if self.transform:
                positive_image = self.transform(positive_image)

            # Negative: different class
            negative_label = random.choice([
                l for l in self.class_to_images.keys()
                if l != anchor_label
            ])
            negative_image_path = random.choice(self.class_to_images[negative_label])
            negative_image = Image.open(negative_image_path).convert("RGB")
            if self.transform:
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image

def get_image_dataloader(image_dir, batch_size=32, shuffle=True, metric_learning=False):
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = LungCancerDataset(
        image_dir=image_dir,
        transform=image_transform,
        metric_learning=metric_learning
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=4
    )
    return dataloader, dataset

class InsuranceDataset(Dataset):
    def __init__(self, csv_path, classification=True, transform=None):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.classification = classification

        # Convert text columns => numeric
        self.data["sex"] = self.data["sex"].map({"female":0, "male":1})
        self.data["smoker"] = self.data["smoker"].map({"no":0, "yes":1})
        region_map = {"southwest":0, "southeast":1, "northwest":2, "northeast":3}
        self.data["region"] = self.data["region"].map(region_map)

        if self.classification:
            # Bin 'charges' => e.g. 3 classes: [0..10000, 10000..20000, 20000+]
            self.data["target"] = pd.cut(
                self.data["charges"],
                bins=[0, 10000, 20000, float("inf")],
                labels=[0, 1, 2]
            )
            self.data.dropna(subset=["target"], inplace=True)
            self.data["target"] = self.data["target"].astype(int)
        else:
            # for regression
            self.data["target"] = self.data["charges"].astype(float)

        self.feature_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
        self.X = self.data[self.feature_cols].values
        self.y = self.data["target"].values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self.classification:
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32)

        return X_tensor, y_tensor

def get_tabular_dataloader(csv_path, batch_size=32, shuffle=True, classification=True):
    dataset = InsuranceDataset(
        csv_path=csv_path,
        classification=classification,
        transform=None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return dataloader, dataset
