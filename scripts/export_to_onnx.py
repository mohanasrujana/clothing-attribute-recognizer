import torch
import torchvision.models as models
import torch.nn as nn
import onnx
from torchvision import transforms
from PIL import Image

#  number of clothing categories and attributes based on the dataset)
num_categories = 50  
num_attributes = 1000 

# Define a modified ResNet-18 model with two output heads
class ResNet18WithAttributes(nn.Module):
    def __init__(self, num_categories, num_attributes):
        """
        Initializes the modified ResNet-18 model with two output heads: one for clothing category prediction
        and one for attribute prediction (multi-label classification).

        Args:
            num_categories (int): The number of clothing categories (e.g., 50 clothing types).
            num_attributes (int): The number of clothing attributes (e.g., 1000 attributes like color, material).
        """
        super(ResNet18WithAttributes, self).__init__()
        self.resnet18 = models.resnet18(weights='IMAGENET1K_V1')
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()  # type: ignore 

        # Define two output heads
        self.category_head = nn.Linear(in_features, num_categories)
        self.attribute_head = nn.Linear(in_features, num_attributes)

    def forward(self, x):
        """
        Forward pass through the network. The image is passed through ResNet-18 to extract features,
        then the features are used to predict both the clothing category and the attributes.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 3, 224, 224) representing the image.

        Returns:
            torch.Tensor: category_logits, attribute_logits - Logits for category and attribute predictions.
        """
        features = self.resnet18(x)
        category_logits = self.category_head(features)
        attribute_logits = self.attribute_head(features)
        return category_logits, attribute_logits

# Create the modified model
model = ResNet18WithAttributes(num_categories, num_attributes)
model.eval()

# Define a dummy input tensor (Batch size: 1, Channels: 3, Image size: 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX format
onnx_model_path = "models/resnet_clothing_with_attributes.onnx"
torch.onnx.export(
    model, 
    dummy_input,  # type: ignore
    onnx_model_path, 
    export_params=True,  
    opset_version=11,   
    do_constant_folding=True,  
    input_names=["input"],  
    output_names=["category_output", "attribute_output"],  
    dynamic_axes={
        "input": {0: "batch_size"},  
        "category_output": {0: "batch_size"}, 
        "attribute_output": {0: "batch_size"} 
    }
)

print(f"Model exported to ONNX: {onnx_model_path}")

# Optional: Test the ONNX model with a real image
def test_onnx_model(onnx_model_path, image_path):
    """
    Test the exported ONNX model by performing inference on a real image.

    This function loads the ONNX model, applies the necessary image transformations,
    performs inference to predict both the clothing category and attributes, and outputs the prediction results.

    Args:
        onnx_model_path (str): Path to the ONNX model.
        image_path (str): Path to the image to test the model with.
    
    Returns:
        None
    """
    import onnxruntime as ort
    import numpy as np

    # Load the ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Define the preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and preprocess an image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).numpy()  # type: ignore 

    # Perform inference
    outputs = ort_session.run(None, {"input": input_tensor})
    category_logits, attribute_logits = outputs

    # Convert logits to probabilities
    category_probs = np.exp(category_logits) / np.sum(np.exp(category_logits), axis=1, keepdims=True) 
    attribute_probs = 1 / (1 + np.exp(-attribute_logits)) 

    # Print the results
    print(f"Category Probabilities: {category_probs}")
    print(f"Attribute Probabilities: {attribute_probs}")

# Test the ONNX model with a sample image
test_onnx_model(onnx_model_path, "dataset/img/2-in-1_Space_Dye_Athletic_Tank/img_00000002.jpg")