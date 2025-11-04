import cv2
import sys
import os
sys.path.append('.')

# Initialize modules
import modules.globals
modules.globals.headless = True
modules.globals.execution_providers = ['CPUExecutionProvider']
modules.globals.frame_processors = ['face_swapper']

from modules.processors.frame.face_swapper import get_face_swapper, process_frame
from modules.face_analyser import get_one_face
from modules.utilities import is_video, extract_frames, create_video, restore_audio, get_temp_frame_paths, create_temp, clean_temp

# Initialize face swapper
face_swapper = get_face_swapper()

def face_swap(source_path, target_path, output_path, mouth_mask=False, many_faces=False):
    """
    Perform face swap on image or video
    
    Args:
        source_path: Path to source face image
        target_path: Path to target image/video
        output_path: Path for output file
        mouth_mask: Preserve original mouth (default: False)
        many_faces: Process all faces (default: False)
    """
    
    # Set global options
    modules.globals.mouth_mask = mouth_mask
    modules.globals.many_faces = many_faces
    modules.globals.source_path = source_path
    modules.globals.target_path = target_path
    modules.globals.output_path = output_path
    
    # Read source image
    source_img = cv2.imread(source_path)
    if source_img is None:
        raise ValueError("Could not read source image")
    
    # Get source face
    source_face = get_one_face(source_img)
    if source_face is None:
        raise ValueError("No face detected in source image")
    
    # Check if target is video or image
    if is_video(target_path):
        # Video processing
        print("Processing video...")
        
        # Create temp directory for frames
        create_temp(target_path)
        extract_frames(target_path)
        
        # Process each frame
        temp_frame_paths = get_temp_frame_paths(target_path)
        
        for i, frame_path in enumerate(temp_frame_paths):
            frame = cv2.imread(frame_path)
            result_frame = process_frame(source_face, frame)
            cv2.imwrite(frame_path, result_frame)
            
            if i % 30 == 0:  # Progress every 30 frames
                print(f"Processed {i+1}/{len(temp_frame_paths)} frames")
        
        # Create output video
        create_video(target_path)
        restore_audio(target_path, output_path)
        
        # Clean up
        clean_temp(target_path)
        
        print(f"Video face swap completed: {output_path}")
        
    else:
        # Image processing
        target_img = cv2.imread(target_path)
        if target_img is None:
            raise ValueError("Could not read target image")
        
        # Process frame
        result_frame = process_frame(source_face, target_img)
        if result_frame is None:
            raise ValueError("Face swap failed - no face detected in target image")
        
        # Save result
        cv2.imwrite(output_path, result_frame)
        print(f"Image face swap completed: {output_path}")

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual file paths
    source_image = "source_face.jpg"
    target_media = "target_video.mp4"  # or "target_image.jpg"
    output_file = "result.mp4"  # or "result.jpg"
    
    try:
        # Basic face swap
        face_swap(source_image, target_media, output_file)
        
        # Face swap with mouth mask (preserves original mouth)
        # face_swap(source_image, target_media, output_file, mouth_mask=True)
        
        # Face swap processing all faces
        # face_swap(source_image, target_media, output_file, many_faces=True)
        
        # Face swap with both options
        # face_swap(source_image, target_media, output_file, mouth_mask=True, many_faces=True)
        
    except Exception as e:
        print(f"Error: {e}")