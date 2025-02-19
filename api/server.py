import torch
import onnxruntime as ort
import csv
import warnings
from typing import Any
import torchvision.transforms as transforms
from PIL import Image
from pydantic import BaseModel
from flask import jsonify, request
from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (ResponseBody, DirectoryInput, FileResponse, FileType,
    InputSchema, ParameterSchema, InputType, EnumParameterDescriptor, EnumVal, TaskSchema
)
from pathlib import Path

warnings.filterwarnings("ignore")

# Initialize Flask-ML Server
server = MLServer(__name__)

# Load ONNX Model
onnx_model_path = "models/resnet_clothing_with_attributes.onnx"
ort_session = ort.InferenceSession(onnx_model_path)

# Define the number of clothing categories and attributes
num_categories = 50  
num_attributes = 1000  

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

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define input and parameter types
class ClothingInputs(TypedDict):
    """
    Defines the expected input structure for the clothing recognition task.

    Attributes:
        input_dataset (DirectoryInput): Path to the directory containing images.
        output_file (DirectoryInput): Path to store the output file.
    """
    input_dataset: DirectoryInput
    output_file: DirectoryInput

class ClothingParameters(TypedDict):
    """
    Defines the parameters used for prediction.

    Attributes:
        threshold (str): Confidence threshold for attribute classification.
    """
    threshold: str 

class PredictionResponse(BaseModel):
    """
    Represents the API response for a prediction request.

    Attributes:
        predicted_category (str): The predicted clothing category.
        most_probable_attribute (dict): Dictionary containing the most probable attribute and its confidence score.
        status (str): Status message indicating success or failure.
    """
    predicted_category: str
    most_probable_attribute: dict
    status: str

# Define the UI schema for the task
def create_clothing_task_schema() -> TaskSchema:
    """
    Creates a schema for the clothing recognition task to define UI input fields and parameters.

    Returns:
        TaskSchema: A schema defining the required inputs and configurable parameters.
    """
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing images",
        input_type=InputType.DIRECTORY,

    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    parameter_schema = ParameterSchema(
        key="threshold",
        label="Confidence Threshold",
        subtitle="Minimum confidence score for attribute prediction.",
        value=EnumParameterDescriptor(
            enum_vals=[
                EnumVal(key="0.5", label="0.5 (Default)"),  
                EnumVal(key="0.7", label="0.7 (High Confidence)"),
                EnumVal(key="0.9", label="0.9 (Very High Confidence)"),
            ],
            default="0.5"  # Default value as a string
        )
    )
    return TaskSchema(
        inputs=[input_schema,output_schema],
        parameters=[parameter_schema]
    )

@server.route("/predict", task_schema_func=create_clothing_task_schema, short_title="Clothing Recognition", order=0)
def predict(inputs: ClothingInputs, parameters: ClothingParameters) -> ResponseBody:
    """
    Process a batch of images from the input directory, perform clothing category and attribute 
    recognition using a pretrained ONNX model, and save predictions to a CSV file.

    Args:
        inputs (ClothingInputs): Dictionary containing paths to the input image directory and output directory.
        parameters (ClothingParameters): Dictionary containing additional model parameters (e.g., confidence threshold).

    Returns:
        ResponseBody: A response containing the path to the generated CSV file with predictions.
    """
    input_dir = Path(inputs["input_dataset"].path)
    output_dir = Path(inputs["output_file"].path)
    output_file = output_dir / "predictions.csv"
    results = []
    for image_path in input_dir.glob("*.jpg"):
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).numpy()
            category_logits, attribute_logits = ort_session.run(None, {"input": input_tensor})

            category_probs = torch.softmax(torch.tensor(category_logits), dim=1).squeeze().tolist()
            attribute_probs = torch.sigmoid(torch.tensor(attribute_logits)).squeeze().tolist()

            predicted_category = category_labels[category_probs.index(max(category_probs))]
            most_probable_attribute = attribute_labels[attribute_probs.index(max(attribute_probs))]

            results.append({
                "image_path": str(image_path),
                "predicted_category": predicted_category,
                "most_probable_attribute": most_probable_attribute,
                "confidence": max(category_probs)
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "predicted_category", "most_probable_attribute", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    return ResponseBody(FileResponse(path=str(output_file), file_type=FileType.CSV))
       
# Add application metadata
server.add_app_metadata(
    name="Clothing Category and Attribute Recognizer",
    author="Satya Srujana Pilli",
    version="1.0.0",
    info="This application recognizes clothing categories and attributes from images."
)

@server.app.route("/", methods=["GET"])
def root():
    """
    Root endpoint to verify that the server is running.
    
    Returns:
        str: A welcome message.
    """
    return "Welcome to the Clothing Attribute Recognizer!"

@server.app.route("/test-upload", methods=["POST"])
def test_upload():
    """
    Test endpoint for uploading an image file.

    This endpoint is used to verify that file uploads are functioning correctly.

    Returns:
        Tuple[str, int]: A success message with HTTP status 200 if the file is uploaded, 
                         otherwise an error message with status 400.
    """
    if "image" not in request.files:
        return "No file uploaded", 400
    file = request.files["image"]
    return f"File {file.filename} uploaded successfully", 200

# Run the server
if __name__ == "__main__":
    print("Starting Flask-ML server...")
    server.run(port=5000)  
    print("Server is running.")