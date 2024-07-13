import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import torch.nn.functional as F
from PIL import Image
import io
import base64
import json
import logging

# Configurar logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
        self.model.classifier[3] = nn.Linear(num_ftrs, 14)
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
            
# Cargar el modelo
def model_fn(model_dir):
    """
    Load the model from the specified directory.
    Args:
        model_dir: Directory containing the model file
    Returns:
        MobileNetV3: Loaded model
    """
    logger.info(f"Loading model from {model_dir}")
    model = MobileNetV3()
    state_dict = torch.load(f"{model_dir}/model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded and set to evaluation mode")
    return model

def pad_to_square_and_resize(img, size=(224, 224)):
    """
    Pads the image to make it square and then resizes it.
    Args:
        img: Input PIL image
        size: Target size
    Returns:
        PIL Image: Transformed image
    """
    max_dim = max(img.size)
    padding = [(max_dim - s) // 2 for s in img.size]
    transform = transforms.Compose([
        transforms.Pad(padding, fill=255),
        transforms.Resize(size)
    ])
    return transform(img)
    
# Función de transformación de imagen
def input_fn(request_body, request_content_type):
    """
    Transform the input image for model inference.
    Args:
        request_body: Input data
        request_content_type: Content type of the input data
    Returns:
        Tensor: Transformed image tensor
    """
    logger.info(f"Received request with content type: {request_content_type}")
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        image_data = request['instances'][0]['image_bytes']['b64']
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    elif request_content_type in ['image/jpeg', 'image/png']:
        image = Image.open(io.BytesIO(request_body))
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

    transform = transforms.Compose([
        transforms.Lambda(pad_to_square_and_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Añadir la dimensión de batch
    logger.info("Image transformed successfully")
    return image

# Realizar la inferencia
def predict_fn(input_data, model):
    """
    Perform inference on the input data.
    Args:
        input_data: Transformed input data
        model: Loaded model
    Returns:
        Tensor: Model predictions
    """
    logger.info("Performing inference")
    with torch.no_grad():
        output = model(input_data)
    logger.info("Inference completed successfully")
    return output

# Formatear la respuesta
def output_fn(prediction, response_content_type):
    """
    Format the prediction output.
    Args:
        prediction: Model predictions
        response_content_type: Desired content type for the response
    Returns:
        str: JSON-formatted prediction output
    """
    logger.info(f"Preparing response with content type: {response_content_type}")
    if response_content_type == 'application/json':
        result = prediction.cpu().numpy().tolist()
        return json.dumps(result)
    else:
        logger.error(f"Unsupported response content type: {response_content_type}")
        raise ValueError(f"Unsupported response content type: {response_content_type}")