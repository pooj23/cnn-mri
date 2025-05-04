import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {
            'brain_glioma': 'glioma',
            'brain_menin': 'meningioma',
            'brain_tumor': 'pituitary'
        }
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes.values())}
        
        # Get all image paths
        self.image_paths = []
        self.labels = []
        
        for cls_dir, cls_name in self.classes.items():
            full_path = os.path.join(root_dir, 'Brain_Cancer raw MRI data', 'Brain_Cancer', cls_dir)
            if os.path.exists(full_path):
                for img_name in os.listdir(full_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(full_path, img_name))
                        self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_loaders(root_dir, batch_size=32):
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = BrainMRIDataset(root_dir, transform=transform)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 