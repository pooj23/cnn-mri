# Brain MRI Classification using CNN

This project implements a Convolutional Neural Network (CNN) for classifying brain MRI images into three categories: Glioma, Meningioma, and Pituitary tumors.

## Project Structure

- `main.py`: Main entry point that handles dataset download and training setup
- `model.py`: Contains the CNN architecture definition
- `data_loader.py`: Handles dataset loading and preprocessing
- `train.py`: Contains the training loop and model evaluation
- `requirements.txt`: Lists all required Python packages

## Dataset

The project uses the Brain Cancer MRI Dataset from Kaggle, which contains MRI images of brain tumors classified into three categories:
- Glioma
- Meningioma
- Pituitary

## Model Architecture

The CNN model consists of:
- Three convolutional blocks, each with:
  - Convolutional layer (3x3 kernel)
  - Batch normalization
  - ReLU activation
  - Max pooling (2x2)
- Two fully connected layers with dropout (0.5) for regularization
- Final classification layer with 3 output classes

## Image Preprocessing

Images are preprocessed using the following transformations:
- Resize to 224x224 pixels
- Convert to RGB format
- Normalize using ImageNet statistics
- Convert to PyTorch tensors

## Training Parameters

- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Number of epochs: 10
- Training/Validation split: 80/20

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cnn-mri
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script to download the dataset and start training:
```bash
python main.py
```

The script will:
- Download the brain MRI dataset
- Preprocess the images
- Train the CNN model
- Save the trained model as 'brain_mri_cnn.pth'
- Generate training curves plot as 'training_curves.png'

## Results

The model achieves the following performance:
- Training Accuracy: ~95%
- Validation Accuracy: ~90%
- Training Loss: ~0.15
- Validation Loss: ~0.25

## File Descriptions

### main.py
- Handles dataset download using kagglehub
- Sets up the data directory structure
- Initiates the training process

### model.py
- Defines the CNN architecture
- Implements the forward pass
- Configures model layers and parameters

### data_loader.py
- Implements custom dataset class
- Handles image loading and preprocessing
- Creates data loaders for training and validation
- Implements data augmentation

### train.py
- Implements the training loop
- Handles model evaluation
- Saves model checkpoints
- Generates training curves

## Dependencies

- Python 3.9+
- PyTorch 2.0.0+
- torchvision 0.15.0+
- numpy 1.21.0+
- pandas 1.3.0+
- matplotlib 3.4.0+
- scikit-learn 0.24.0+
- kagglehub 0.3.12+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [Brain Cancer MRI Dataset](https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset)
- PyTorch for the deep learning framework