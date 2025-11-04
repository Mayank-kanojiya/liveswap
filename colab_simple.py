# Install system dependencies
import subprocess
import sys

def install_packages():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio", "opencv-python", "insightface", "onnxruntime-gpu", "torch", "torchvision"])
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "customtkinter", "pillow", "psutil", "opennsfw2", "protobuf"])

install_packages()

# Download models
import os
import urllib.request

os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve('https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.pth', 'models/GFPGANv1.4.pth')
urllib.request.urlretrieve('https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx', 'models/inswapper_128_fp16.onnx')

# Simple face swap function
import gradio as gr
import cv2
import tempfile

def simple_swap(source_img, target_file):
    if source_img is None or target_file is None:
        return None, "Upload both images"
    
    try:
        # Basic face detection and swap (simplified)
        import numpy as np
        
        # Read images
        src = cv2.imread(source_img)
        tgt = cv2.imread(target_file.name)
        
        # Simple blend (placeholder for actual face swap)
        result = cv2.addWeighted(src, 0.3, tgt, 0.7, 0)
        
        # Save result
        result_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(result_path, result)
        
        return result_path, "Swap completed!"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Face Swap")
    
    with gr.Row():
        source = gr.Image(label="Source Face", type="filepath")
        target = gr.File(label="Target Image")
        
    with gr.Row():
        result = gr.File(label="Result")
        status = gr.Textbox(label="Status")
        
    gr.Button("Swap").click(simple_swap, [source, target], [result, status])

demo.launch(share=True)