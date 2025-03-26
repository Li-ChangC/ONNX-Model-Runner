# ONNX Model Runner

This is a simple yet powerful tool that demonstrates how to convert a PyTorch model to ONNX format and run inference using ONNX Runtime. This project serves as a practical example for deploying machine learning models in a lightweight and production-ready format.

## Features

- ✨ PyTorch to ONNX model conversion  
- ⚖️ ONNX Runtime-based inference  
- 📊 Confidence-based filtering and Non-Maximum Suppression (NMS)  
- 📷 Easy testing on sample images  

## Project Structure
```
ONNX-Model-Runner/
├── pt_to_onnx.py             # Convert PyTorch .pt model to ONNX format
├── run_on_onnxruntime.py     # Run inference using ONNX Runtime
├── class_names.py            # Class label definitions
├── TestFiles/                # Test images
│   └──sample.jpg
└── models/                   # Sample models
    ├── sample.pt
    └── sample.onnx
```

## Note
This uses a sample model to classify images. If you're going to use other models, code in `run_on_onnxruntime.py` should be modified to match the architecture accordingly.