import torch
from model import BrainMRICNN
from torchvision import transforms
from PIL import Image
import os

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(model_path='brain_mri_cnn.pth'):
    device = get_device()
    model = BrainMRICNN(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict_image(model, device, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0].tolist()

def main():
    # Load the trained model
    model, device = load_model()
    
    # Class names
    classes = ['glioma', 'meningioma', 'pituitary']
    
    # Example usage
    test_image_path = 'path_to_your_test_image.jpg'  # Replace with actual image path
    
    if os.path.exists(test_image_path):
        # Preprocess the image
        image_tensor = preprocess_image(test_image_path)
        
        # Make prediction
        predicted_class, probabilities = predict_image(model, device, image_tensor)
        
        # Print results
        print(f"\nPredicted class: {classes[predicted_class]}")
        print("\nClass probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{classes[i]}: {prob:.4f}")
    else:
        print(f"Error: Image not found at {test_image_path}")

if __name__ == '__main__':
    main() 