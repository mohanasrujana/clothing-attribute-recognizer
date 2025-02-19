import onnx

# Load the ONNX model
onnx_model = onnx.load("models/resnet_clothing_with_attributes.onnx")

# Verify the model
onnx.checker.check_model(onnx_model)

print("ONNX model is valid!")