# Simple script using original Deep-Live-Cam mouth mask feature
import sys
import os
sys.path.append('.')

# Use original mouth mask feature
import modules.globals
modules.globals.headless = True
modules.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
modules.globals.frame_processors = ['face_swapper']
modules.globals.mouth_mask = True  # Enable original mouth mask
modules.globals.many_faces = False

from modules.processors.frame.face_swapper import get_face_swapper, process_frame
from modules.face_analyser import get_one_face
from modules.utilities import is_video, extract_frames, create_video, restore_audio, get_temp_frame_paths, create_temp, clean_temp

# Initialize
face_swapper = get_face_swapper()

def simple_face_swap_with_mouth_mask(source_path, target_path, output_path):
    """Use original Deep-Live-Cam mouth mask feature"""
    
    modules.globals.source_path = source_path
    modules.globals.target_path = target_path
    modules.globals.output_path = output_path
    
    source_img = cv2.imread(source_path)
    source_face = get_one_face(source_img)
    
    if is_video(target_path):
        print("Processing video with original mouth mask...")
        create_temp(target_path)
        extract_frames(target_path)
        
        temp_frame_paths = get_temp_frame_paths(target_path)
        
        for i, frame_path in enumerate(temp_frame_paths):
            frame = cv2.imread(frame_path)
            # Original mouth mask is handled internally by process_frame
            result_frame = process_frame(source_face, frame)
            cv2.imwrite(frame_path, result_frame)
            
            if i % 30 == 0:
                print(f"Processed {i+1}/{len(temp_frame_paths)} frames")
        
        create_video(target_path)
        restore_audio(target_path, output_path)
        clean_temp(target_path)
        
        # Check if file exists and copy to expected location
        temp_output = f"/tmp/{os.path.basename(target_path)}"
        if os.path.exists(temp_output):
            import shutil
            shutil.copy2(temp_output, output_path)
        
    else:
        target_img = cv2.imread(target_path)
        result_frame = process_frame(source_face, target_img)
        cv2.imwrite(output_path, result_frame)
    
    print(f"Face swap with mouth mask completed: {output_path}")

# Usage
if __name__ == "__main__":
    import cv2
    
    source = "/content/source.jpg"
    target = "/content/eating_video.mp4"
    output = "/content/result_mouth_mask.mp4"
    
    simple_face_swap_with_mouth_mask(source, target, output)
    
    # List and download
    print("Available files:")
    for f in os.listdir('/content'):
        if f.endswith('.mp4'):
            print(f"- {f}")
    
    if os.path.exists(output):
        print(f"Success! Output saved: {output}")
    else:
        print("Checking alternative locations...")
        for root, dirs, files in os.walk('/content'):
            for file in files:
                if file.endswith('.mp4') and 'result' in file.lower():
                    full_path = os.path.join(root, file)
                    print(f"Found result: {full_path}")