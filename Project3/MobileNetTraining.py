import os
import torch
import zipfile
import logging
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def extract_zip_from_directory(directory, extract_to):
    """
    Extracts the first zip file found in the specified directory to a target directory.
    Args:
        directory (str): The directory to search for the zip file.
        extract_to (str): The directory where the contents of the zip file will be extracted.
    """
    zip_files = [f for f in os.listdir(directory) if f.endswith('.zip')]
    if not zip_files:
        raise FileNotFoundError("No zip file found in the directory.")
    
    zip_path = os.path.join(directory, zip_files[0])
    logger.debug(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logger.debug(f"Data extracted to {extract_to}")

def pad_to_square(img, max_height, max_width):
    """
    Pads an image to a square with specified maximum dimensions.
    Args:
        img (PIL.Image): The image to be padded.
        max_height (int): The maximum height for the padded image.
        max_width (int): The maximum width for the padded image.
    Returns:
        PIL.Image: The padded image.
    """
    width, height = img.size
    padding = (0, 0, max_width - width, max_height - height)
    return transforms.functional.pad(img, padding, fill=255)

def get_dataloaders(data_directory, batch_size):
    """
    Creates and returns PyTorch DataLoaders for training and testing datasets with specified transformations.
    Args:
        data_directory (str): The directory where the training and testing datasets are stored.
        batch_size (int): The number of samples per batch to load.
    Returns:
        tuple: A tuple containing the training DataLoader and testing DataLoader.
    """
    max_height, max_width = 400, 400
    img_height, img_width = 224, 224

    training_transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_square(img, max_height, max_width)),
        transforms.Resize((img_height, img_width)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop((img_height, img_width), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testing_transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_square(img, max_height, max_width)),
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    trainset = ImageFolder(os.path.join(data_directory, 'train'), transform=training_transform)
    testset = ImageFolder(os.path.join(data_directory, 'test'), transform=testing_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

class MobileNetV3(nn.Module):
    """
    A custom MobileNetV3 model class for transfer learning.
    Methods:
        forward(x): Defines the forward pass of the model.
        freeze(): Freezes all layers except the final classifier layer.
        unfreeze(): Unfreezes all layers.
    """
    def __init__(self):
        super(MobileNetV3, self).__init__()
        self.model = mobilenet_v3_small(weights="IMAGENET1K_V1")
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, 20)
        self.freeze()

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier[3].parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

def evaluate(model, testloader, device):
    """
    Evaluates the performance of a model on a test dataset.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which the model and data are located.
    Returns:
        tuple: A tuple containing the accuracy (as a percentage) and the weighted F1 score of the model on the test dataset.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred) * 100 
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, f1

def save_model(model, model_dir):
    """
    Saves the model's state dictionary to a specified directory.
    Args:
        model (torch.nn.Module): The model to be saved.
        model_dir (str): The directory where the model will be saved.
    """
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)
    logger.debug(f"Model saved to {path}")

def train(args):
    """
    Trains a MobileNetV3 model using the specified hyperparameters and dataset.
    Args:
        args (Namespace): A namespace containing the training hyperparameters such as batch_size, epochs, and lr.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/opt/ml/input/data/train'
    extract_to = '/opt/ml/input/data/extracted'
    os.makedirs(extract_to, exist_ok=True)
    extract_zip_from_directory(data_dir, extract_to)
    trainloader, testloader = get_dataloaders(extract_to, args.batch_size)
    model = MobileNetV3().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.model.classifier[3].parameters(), lr=args.lr)
    logger.debug(f"Hyperparameters: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        for batch_num, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_num % 100 == 0:
                logger.debug(f"Epoch [{epoch+1}/{args.epochs}], Batch [{batch_num}/{len(trainloader)}], Loss: {loss.item():.4f}")
        
        accuracy, f1 = evaluate(model, testloader, device)
        logger.debug(f"Validation Accuracy: {accuracy:.2f}%")
        logger.debug(f"Validation F1-Score (weighted): {f1:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    train(args)