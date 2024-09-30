import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Data Set Class used for loading the Lung Cancer Dataset
class LungCancerDataset(Dataset):
    def __init__(self, image_dir, transform=None, metric_learning=False):
        self.image_dir = image_dir
        self.transform = transform
        self.metric_learning = metric_learning
        
        # Create a class map
        self.class_map = {folder_name: idx for idx, folder_name in enumerate(os.listdir(image_dir)) if os.path.isdir(os.path.join(image_dir, folder_name))}

        self.image_paths = []
        self.labels = []
        self.class_to_images = {label: [] for label in self.class_map.values()}

        for class_name, label in self.class_map.items():
            class_dir = os.path.join(image_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
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
            return image, label
        else:
            # Prepare a triplet (anchor, positive, negative) for metric learning
            # Anchor
            anchor_image = image
            anchor_label = label

            # Positive: Randomly select another image of the same class
            positive_image_path = random.choice(self.class_to_images[anchor_label])
            positive_image = Image.open(positive_image_path).convert("RGB")
            if self.transform:
                positive_image = self.transform(positive_image)

            # Negative: Randomly select an image from a different class
            negative_label = random.choice([l for l in self.class_to_images.keys() if l != anchor_label])
            negative_image_path = random.choice(self.class_to_images[negative_label])
            negative_image = Image.open(negative_image_path).convert("RGB")
            if self.transform:
                negative_image = self.transform(negative_image)

            return anchor_image, positive_image, negative_image


def get_dataloader(image_dir, batch_size=32, shuffle=True, metric_learning=False):
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create an instance of the LungCancerDataset
    dataset = LungCancerDataset(image_dir=image_dir, transform=image_transform, metric_learning=metric_learning)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
