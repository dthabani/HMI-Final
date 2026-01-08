import numpy as np
from deepface import DeepFace

class EmotionManager:
    def __init__(self):
        """
        Initializes the EmotionManager.
        Preload the model to avoid lag on the first trigger.
        """
        print("EmotionManager: Loading DeepFace model (this may take a moment)...")
        # Run a dummy prediction to load the weights into memory.
        try:
            # Create a black dummy image
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            DeepFace.analyze(img_path=dummy_img, actions=['emotion'], 
                        enforce_detection=False, detector_backend='opencv', silent=True)
            print("EmotionManager: Model Loaded & Ready.")
        except Exception as e:
            print(f"EmotionManager: Warning - Model pre-load failed: {e}")

    def detect_emotion(self, frame):
        """
        Runs analysis on a single frame.
        Returns the dominant emotion (str) or None if failed.
        """
        try:
            # DeepFace expects BGR (OpenCV default) or RGB. 
            objs = DeepFace.analyze(img_path=frame, actions=['emotion'], 
                                enforce_detection=False, detector_backend='opencv', silent=True)
            
            if objs:
                # Get dominant emotion
                emotion = objs[0]['dominant_emotion']
                return emotion
            return None
        except Exception as e:
            print(f"EmotionManager Error: {e}")
            return None
