import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import gradio as gr
from model import resnet50  # Import the model definition

# Load class index mapping
def load_class_labels(label_path):
    with open(label_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

# Load the model
def load_model(model_path):
    model = resnet50()  # Create an instance of the ResNet-50 model
    state_dict = torch.load(model_path, map_location='cpu')  # Load the state dictionary

    # Remove "module." prefix if present
    if 'module.' in next(iter(state_dict.keys())):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = image.convert("RGB")  # Ensure image is in RGB format
    image = preprocess(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict(image):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top5_probabilities, top5_indices = probabilities.topk(5)
    
    results = {}
    for i in range(5):
        class_index = top5_indices[0][i].item()
        class_label = class_labels.get(str(class_index), "Unknown class")
        results[class_label] = top5_probabilities[0][i].item()  # Store label and probability in a dictionary
    
    return results  # Return the results as a dictionary

# Load model and class labels
model_path = 'model.pt'  # Path to the trained model
label_path = 'imagenet_class_index.json'  # Path to the class index mapping
model = load_model(model_path)
class_labels = load_class_labels(label_path)

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="Image Classification using ResNet-50 Model",
    description="Upload an image to get the top-5 predictions from the ResNet-50 model."
)

# Launch the app
iface.launch() 