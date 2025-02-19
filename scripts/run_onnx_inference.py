import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the ONNX model
onnx_model_path = "models/resnet_clothing_with_attributes.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load category and attribute labels
def load_labels(file_path):
    """
    Reads a label file and extracts the labels.

    Args:
        file_path (str): Path to the label file.

    Returns:
        list: A list of label names extracted from the file.
    
    Assumption:
        - The first two lines of the file contain metadata and are skipped.
        - Each subsequent line contains a label, extracted from the first word.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    return [line.split()[0] for line in lines[2:]]  

category_labels = load_labels("dataset/list_category_cloth.txt")
attribute_labels = load_labels("dataset/list_attr_cloth.txt")

# Load and preprocess an image
image_path = "dataset/img/2-in-1_Space_Dye_Athletic_Tank/img_00000002.jpg"  # Replace with your image path
image = Image.open(image_path).convert("RGB")  
input_tensor = transform(image).unsqueeze(0)  # type: ignore 
input_tensor = input_tensor.numpy() 

# Perform inference
outputs = ort_session.run(
    None, 
    {"input": input_tensor}  
)

# Extract the outputs
category_logits = outputs[0] 
attribute_logits = outputs[1]  

# Convert logits to probabilities
category_probs = np.exp(category_logits) / np.sum(np.exp(category_logits), axis=1, keepdims=True)
attribute_probs = 1 / (1 + np.exp(-attribute_logits)) 

# Get the predicted category
predicted_category_index = np.argmax(category_probs, axis=1)[0]
predicted_category = category_labels[predicted_category_index]  

# Get the most probable attribute
most_probable_attribute_index = np.argmax(attribute_probs, axis=1)[0]
most_probable_attribute = attribute_labels[most_probable_attribute_index]  
most_probable_attribute_score = attribute_probs[0][most_probable_attribute_index] 

# Print the results
print(f"Predicted Clothing Category for {image_path}: {predicted_category}")
print(f"Most Probable Attribute: {most_probable_attribute} ({most_probable_attribute_score:.2f})")

# Display the image with the predicted category and most probable attribute
plt.imshow(image)
plt.title(f"Predicted Category: {predicted_category} \nMost Probable Attribute: {most_probable_attribute} ({most_probable_attribute_score:.2f})")
plt.axis("off")
plt.show()