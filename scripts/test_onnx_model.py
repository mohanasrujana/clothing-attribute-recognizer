import onnx

# Load the ONNX model
onnx_model = onnx.load("models/resnet_clothing.onnx")

# Verify the model
onnx.checker.check_model(onnx_model)

print("ONNX model is valid!")