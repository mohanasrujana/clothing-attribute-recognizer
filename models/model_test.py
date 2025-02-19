import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels 
def load_labels(category_file):
    """
    Load category labels from a file. The file format assumes the first line is the
    number of categories, and subsequent lines contain the category names.
    """
    with open(category_file, "r") as f:
        lines = f.readlines()
    num_categories = int(lines[0].strip())  # First line: number of categories
    categories = [line.split()[0] for line in lines[2:]]  # Skip first two lines and extract category names
    return categories

# Load images and their labels (for category prediction)
def load_image_labels(image_file):
    """
    Load image-to-label mappings. The file format assumes each line contains an image
    name and the corresponding category label (starting from 1).
    """
    image_labels = {}
    with open(image_file, "r") as f:
        lines = f.readlines()
    for line in lines[2:]:
        image_name, category_label = line.split()
        image_labels[image_name] = int(category_label) - 1  # Adjust for 0-based index
    return image_labels

# Number of categories in dataset 
num_categories = 50  

# Load the pre-trained ResNet-18 model
model = models.resnet18(weights='IMAGENET1K_V1')

# Replace the final fully connected layer to match the number of categories
model.fc = nn.Linear(model.fc.in_features, num_categories)
model.eval()

# Set up image transforms (ResNet18 expected input)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load category and attribute labels
category_labels = load_labels("dataset/list_category_cloth.txt")
attribute_labels = load_labels("dataset/list_attr_cloth.txt")

# Path to the image directory
dataset_dir = "dataset/img"
image_paths = []

# Loop through subdirectories to get image paths (limit to 500 images)
for subdir, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            image_paths.append(os.path.join(subdir, file))
            if len(image_paths) >= 500:  # Limit to first 500 images
                break
    if len(image_paths) >= 500:
        break

random_image_path = random.choice(image_paths)  

# Load the image and preprocess it
image = Image.open(random_image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # type: ignore

# Perform inference for category prediction
output = model(input_tensor)
category_probabilities = F.softmax(output, dim=1)

# Print the shape of the output to debug the issue
print(f"Model output shape: {output.shape}")

# Ensure that the predicted index is within bounds
category_predicted_index = category_probabilities.argmax().item()
predicted_attributes = []
most_probable_attribute = None
max_score = -1  
attribute_predictions = torch.randn(len(attribute_labels)).sigmoid().cpu().detach().numpy()  # Random predictions
for i, attr_label in enumerate(attribute_labels):
    if attribute_predictions[i] > 0.5:  # If the prediction score is greater than 0.5, consider it positive
        predicted_attributes.append(f"{attr_label}: Positive")
        if attribute_predictions[i] > max_score:
            max_score = attribute_predictions[i]
            most_probable_attribute = f"{attr_label}: Positive"
    elif attribute_predictions[i] < 0.5 and attribute_predictions[i] > 0:
        predicted_attributes.append(f"{attr_label}: Negative")
        if attribute_predictions[i] > max_score:
            max_score = attribute_predictions[i]
            most_probable_attribute = f"{attr_label}: Negative"
    else:
        predicted_attributes.append(f"{attr_label}: Unknown")
        if attribute_predictions[i] > max_score:
            max_score = attribute_predictions[i]
            most_probable_attribute = f"{attr_label}: Unknown"

# Debugging check for index range
if category_predicted_index < len(category_labels):
    predicted_category = category_labels[category_predicted_index] # type: ignore
    print(f"Predicted Clothing Category for {random_image_path}: {predicted_category}")
    # Display the image with predicted category
    plt.imshow(image)
    plt.title(f"Predicted Category: {predicted_category} \nMost Probable Attribute: {most_probable_attribute}")
    plt.axis("off")
    plt.show()
    
else:
    print(f"Error: predicted index {category_predicted_index} is out of bounds for category labels.")

