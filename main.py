import kagglehub
import os
import shutil
from data_loader import get_data_loaders
from train import main as train_model

def setup_dataset():
    # Download dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("orvile/brain-cancer-mri-dataset")
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Copy the dataset to the data directory
    print("Copying dataset...")
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join('data', item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    
    print("Dataset setup complete!")

if __name__ == "__main__":
    # Setup dataset
    setup_dataset()
    
    # Train the model
    print("Starting model training...")
    train_model()