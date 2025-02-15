import torch
import torchvision.models as models
import torch.nn as nn
import onnx

# Define the number of clothing categories (based on your dataset)
num_categories = 50  # Update this if necessary

# Load pretrained ResNet-18
model = models.resnet18(weights='IMAGENET1K_V1')

# Modify the final layer to match the number of clothing categories
model.fc = nn.Linear(model.fc.in_features, num_categories)

# Set model to evaluation mode
model.eval()

# Define a dummy input tensor (Batch size: 1, Channels: 3, Image size: 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX format
onnx_model_path = "models/resnet_clothing.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_model_path, 
    export_params=True,  # Store model parameters
    opset_version=11,  # Set ONNX opset version
    do_constant_folding=True,  # Optimize constant expressions
    input_names=["input"],  # Define input layer name
    output_names=["output"],  # Define output layer name
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allow batch size flexibility
)

print(f"✅ Model exported to ONNX: {onnx_model_path}")