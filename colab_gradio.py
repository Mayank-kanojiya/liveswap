import gradio as gr
import cv2
import os
import tempfile
from modules.processors.frame.face_swapper import process_frame
from modules.face_analyser import get_one_face
from modules.utilities import is_image, is_video
import modules.globals

def swap_face(source_image, target_media):
    if source_image is None or target_media is None:
        return None, "Please provide both source image and target media"
    
    try:
        # Set source path
        modules.globals.source_path = source_image
        
        # Process based on media type
        if is_image(target_media):
            # Image processing
            result_path = tempfile.mktemp(suffix='.jpg')
            frame = cv2.imread(target_media)
            result_frame = process_frame(get_one_face(cv2.imread(source_image)), frame)
            cv2.imwrite(result_path, result_frame)
            return result_path, "Face swap completed successfully!"
            
        elif is_video(target_media):
            # Video processing
            result_path = tempfile.mktemp(suffix='.mp4')
            cap = cv2.VideoCapture(target_media)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            
            source_face = get_one_face(cv2.imread(source_image))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result_frame = process_frame(source_face, frame)
                out.write(result_frame)
            
            cap.release()
            out.release()
            return result_path, "Video face swap completed successfully!"
            
    except Exception as e:
        return None, f"Error: {str(e)}"

# Gradio interface
with gr.Blocks(title="Deep-Live-Cam") as demo:
    gr.Markdown("# Deep-Live-Cam - Face Swap")
    
    with gr.Row():
        with gr.Column():
            source_image = gr.Image(label="Source Face Image", type="filepath")
            target_media = gr.File(label="Target Image/Video")
            swap_btn = gr.Button("Swap Face", variant="primary")
            
        with gr.Column():
            result_media = gr.File(label="Result")
            status_text = gr.Textbox(label="Status", interactive=False)
    
    swap_btn.click(
        fn=swap_face,
        inputs=[source_image, target_media],
        outputs=[result_media, status_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)