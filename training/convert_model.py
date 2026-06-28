"""
Model Conversion Utility: PyTorch Lightning to ONNX

This script loads a trained neural network checkpoint (.ckpt), sets it to evaluation 
mode, and exports it into an optimized, platform-independent ONNX format (.onnx) 
using a dummy input tensor to trace the mathematical computational graph.
"""

import torch
from CNN import CNN


def load_model(fold_label: str) -> CNN:
    """
    Load the weights of the highest-performing model from a specific cross-validation fold.
    """
    # Build the path targeting the best local checkpoint file
    model_path = f"checkpoints/{fold_label}/best-checkpoint.ckpt"
    
    # Instantiate the architecture and inject the saved mathematical weights.
    # The dimensions match the 3-second window (180 samples) processed during training.
    model = CNN.load_from_checkpoint(
        checkpoint_path=model_path,
        input_dim=180,
        fold=1,  # Reset the initial internal fold counter for baseline evaluations
        classes_names=["balling", "shooting", "sitting", "standing"]
    ).to("cpu")  # Exporting can be safely executed on local CPU context
    
    # Set model to evaluation mode (deactivates dropout and batch normalization updates)
    model.eval()
    return model


def torch_to_onnx(model: CNN, dummy_input: torch.Tensor) -> None:
    """
    Trace the execution graph of the PyTorch model and export it to an ONNX binary file.
    """
    output_filename = f"{model.id()}.onnx"
    
    print(f"Exporting model architecture graph to format: {output_filename}...")
    
    torch.onnx.export(
        model,
        dummy_input,                # Simulated sample vector used by PyTorch to trace internal tensor shapes
        output_filename,            # Target output file path destination
        export_params=True,         # Embed the trained weights directly inside the exported file
        verbose=False,              # Set to True if you need full internal layer matrix tracking logs
        opset_version=10            # Standard cross-platform stable mathematical operations set version
    )
    print("Model conversion completed successfully.")


def main():
    # Targeted cross-validation folder containing the checkpoint to convert
    target_fold = "fold_1"
    
    # Create simulated hardware input data matching the structural dimensions:
    # (Batch size = 1, Sequence temporal frames = 180, Extracted features = 6)
    dummy_tensor = torch.randn(1, 180, 6)
    
    # Execute loading and conversion routines sequentially
    trained_model = load_model(target_fold)
    torch_to_onnx(trained_model, dummy_tensor)


if __name__ == '__main__':
    # This block executes automatically because you will run 'python convert_model.py'
    # as an independent script once your training phase concludes.
    main()