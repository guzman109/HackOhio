import os, sys
import numpy as np

# Add all the repos we cloned to the Python Path 
sys.path.append(f"{os.environ['MODEL_REPO']}/MegaDetector/1/")
for dir in ['ai4eutils', 'camera_traps_MD', 'yolov5']:
    sys.path.append( f"{os.environ['MODEL_REPO']}/MegaDetector/1/{dir}" )


class MegaDetector:
    def __init__(self):
        # Path model directory
        model_path = f"{ os.environ['MODEL_REPO'] }/MegaDetector/1"
        
        # Load MegaDetector
        from run_detector_multi import load_detector
        self.model = load_detector( f"{ model_path }/md_v5a.0.0.pt", force_cpu=True )

    def __call__(self, image):
        detections = self.model.generate_detections_one_image( image, "", 0.005 )

        return detections    
