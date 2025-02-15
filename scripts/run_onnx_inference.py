import onnxruntime as ort
import numpy as np

# Load ONNX model into ONNXRuntime
ort_session = ort.InferenceSession("models/resnet_clothing.onnx")

# Create a dummy input tensor
dummy_input_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = ort_session.run(None, {"input": dummy_input_np})

print(f" ONNX model inference successful! Output shape: {outputs[0].shape}")