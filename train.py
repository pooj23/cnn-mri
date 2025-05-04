import torch
import torch.nn as nn
import torch.optim as optim
from model import BrainMRICNN
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import json

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def plot_confusion_matrix(y_true, y_pred, classes, filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_training_history(train_losses, val_losses, train_accs, val_accs, filename='training_history.json'):
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    with open(filename, 'w') as f:
        json.dump(history, f)

def load_training_history(filename='training_history.json'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None, 
                model_path='brain_mri_cnn.pth', history_path='training_history.json', force_retrain=False):
    if device is None:
        device = get_device()
    
    # Check if model and history exist
    if not force_retrain and os.path.exists(model_path) and os.path.exists(history_path):
        print("Loading cached model and training history...")
        model.load_state_dict(torch.load(model_path))
        history = load_training_history(history_path)
        if history:
            train_losses = history['train_losses']
            val_losses = history['val_losses']
            train_accs = history['train_accs']
            val_accs = history['val_accs']
            return train_losses, val_losses, train_accs, val_accs

    # Move model to device
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                # Move data to device
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save model and history after each epoch
        torch.save(model.state_dict(), model_path)
        save_training_history(train_losses, val_losses, train_accs, val_accs, history_path)
    
    # Plot confusion matrix for the final validation results
    classes = ['glioma', 'meningioma', 'pituitary']
    plot_confusion_matrix(all_labels, all_preds, classes)
    
    return train_losses, val_losses, train_accs, val_accs

def main(force_retrain=False):
    # Set device
    device = get_device()
    print(f'Using device: {device}')
    
    # Create model
    model = BrainMRICNN(num_classes=3)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders('data', batch_size=32)
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=10, device=device, force_retrain=force_retrain
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.savefig('training_curves.png')
    plt.close()

if __name__ == '__main__':
    main() 