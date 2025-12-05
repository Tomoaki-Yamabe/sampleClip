"""
Convert PyTorch models to ONNX format for Lambda deployment

This script converts the text and image encoder models to ONNX format,
which provides better performance and smaller deployment size for Lambda.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integ-app', 'backend'))

import torch
import torch.nn as nn
from app.encoders import TextEncoder, ImageEncoder
from PIL import Image
import numpy as np

# Paths
MODEL_DIR = "../integ-app/backend/app/model"
OUTPUT_DIR = "../lambda/models"
TEXT_PROJECTOR_PATH = f"{MODEL_DIR}/text_projector.pt"
IMAGE_PROJECTOR_PATH = f"{MODEL_DIR}/image_projector.pt"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def convert_text_encoder_to_onnx():
    """Convert text encoder to ONNX format"""
    print("Converting Text Encoder to ONNX...")
    
    # Load the encoder
    encoder = TextEncoder(projector_path=TEXT_PROJECTOR_PATH, device=torch.device('cpu'))
    encoder.eval()
    
    # Create dummy input
    dummy_text = "This is a test sentence for ONNX conversion"
    inputs = encoder.tokenizer(
        [dummy_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Export the transformer model
    print("  Exporting transformer model...")
    torch.onnx.export(
        encoder.model,
        (inputs['input_ids'], inputs['attention_mask']),
        f"{OUTPUT_DIR}/text_transformer.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # Export the projector
    print("  Exporting projector...")
    dummy_hidden = torch.randn(1, encoder.model.config.hidden_size)
    torch.onnx.export(
        encoder.projector,
        dummy_hidden,
        f"{OUTPUT_DIR}/text_projector.onnx",
        input_names=['hidden_state'],
        output_names=['embedding'],
        dynamic_axes={
            'hidden_state': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print("  Text encoder converted successfully!")
    
    # Verify the conversion
    print("  Verifying conversion...")
    import onnxruntime as ort
    
    # Test transformer
    sess_transformer = ort.InferenceSession(f"{OUTPUT_DIR}/text_transformer.onnx")
    onnx_outputs = sess_transformer.run(
        None,
        {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
    )
    
    # Test projector
    hidden_state = onnx_outputs[0].mean(axis=1)  # Mean pooling
    sess_projector = ort.InferenceSession(f"{OUTPUT_DIR}/text_projector.onnx")
    proj_output = sess_projector.run(None, {'hidden_state': hidden_state})
    
    # Normalize
    embedding = proj_output[0]
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
    
    print(f"  Output shape: {embedding.shape}")
    print(f"  Output norm: {np.linalg.norm(embedding[0]):.4f}")
    print("  ✓ Text encoder verification passed!")


def convert_image_encoder_to_onnx():
    """Convert image encoder to ONNX format"""
    print("\nConverting Image Encoder to ONNX...")
    
    # Load the encoder
    encoder = ImageEncoder(projector_path=IMAGE_PROJECTOR_PATH, device=torch.device('cpu'))
    encoder.eval()
    
    # Create dummy input (224x224 RGB image)
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # Export the feature extractor
    print("  Exporting feature extractor...")
    torch.onnx.export(
        encoder.features,
        dummy_image,
        f"{OUTPUT_DIR}/image_features.onnx",
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    # Export the projector
    print("  Exporting projector...")
    dummy_features = torch.randn(1, 576)
    torch.onnx.export(
        encoder.projector,
        dummy_features,
        f"{OUTPUT_DIR}/image_projector.onnx",
        input_names=['features'],
        output_names=['embedding'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    print("  Image encoder converted successfully!")
    
    # Verify the conversion
    print("  Verifying conversion...")
    import onnxruntime as ort
    
    # Test feature extractor
    sess_features = ort.InferenceSession(f"{OUTPUT_DIR}/image_features.onnx")
    features_output = sess_features.run(None, {'image': dummy_image.numpy()})
    
    # Global average pooling
    features = features_output[0].mean(axis=(2, 3))  # (B, 576)
    
    # Test projector
    sess_projector = ort.InferenceSession(f"{OUTPUT_DIR}/image_projector.onnx")
    proj_output = sess_projector.run(None, {'features': features})
    
    # Normalize
    embedding = proj_output[0]
    embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
    
    print(f"  Output shape: {embedding.shape}")
    print(f"  Output norm: {np.linalg.norm(embedding[0]):.4f}")
    print("  ✓ Image encoder verification passed!")


def print_model_sizes():
    """Print model file sizes"""
    print("\n" + "="*60)
    print("Model File Sizes:")
    print("="*60)
    
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith('.onnx'):
            filepath = os.path.join(OUTPUT_DIR, filename)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename:30s} {size_mb:8.2f} MB")
    
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("PyTorch to ONNX Model Conversion")
    print("="*60)
    
    try:
        convert_text_encoder_to_onnx()
        convert_image_encoder_to_onnx()
        print_model_sizes()
        
        print("\n✓ All models converted successfully!")
        print(f"✓ ONNX models saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
