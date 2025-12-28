import cv2
import os
import numpy as np
import json

class FaceAuth:
    def __init__(self, data_path="faces"):
        self.data_path = data_path
        self.model_path = os.path.join(data_path, "lbph_model.yml")
        self.labels_path = os.path.join(data_path, "labels.json")
        
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {} # id -> name
        self.load()

    def load(self):
        """Load model and labels if they exist."""
        try:
            if os.path.exists(self.model_path):
                self.recognizer.read(self.model_path)
                print("Face Auth Model Loaded.")
            
            if os.path.exists(self.labels_path):
                with open(self.labels_path, 'r') as f:
                    self.labels = {int(k): v for k, v in json.load(f).items()}
                print(f"Labels Loaded: {self.labels}")
        except Exception as e:
            print(f"Error loading Face Auth: {e}")

    def save(self):
        """Save model and labels."""
        try:
            self.recognizer.write(self.model_path)
            with open(self.labels_path, 'w') as f:
                json.dump({str(k): v for k, v in self.labels.items()}, f)
            print("Face Auth Model Saved.")
        except Exception as e:
            print(f"Error saving Face Auth: {e}")

    def train(self, images, ids):
        """Train the model with new data."""
        print("Training Face Model...")
        self.recognizer.train(images, np.array(ids))
        self.save()
        print("Training Complete.")

    def predict(self, frame):
        """
        Predict user from frame.
        Returns: (name, confidence)
        """
        if not self.labels:
            return "Unknown", 100.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return "No Face", 100.0
            
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            id_, conf = self.recognizer.predict(roi_gray)
            # Confidence: 0 is perfect match. < 50 is good. > 80 is unknown.
            if conf < 70:
                name = self.labels.get(id_, "Unknown")
                return name, conf
            else:
                return "Unknown", conf
        except Exception as e:
            return "Unknown", 100.0

    def register_new_user(self, frame, name):
        """
        Register user from a single captured frame.
        Generates synthetic samples to help LBPH.
        """
        print(f"Registering: {name}")
        
        # Get new ID
        new_id = 1
        if self.labels:
            new_id = max(self.labels.keys()) + 1
        self.labels[new_id] = name
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("No face detected for registration.")
            return False
            
        (x, y, w, h) = faces[0]
        roi_base = gray[y:y+h, x:x+w]
        
        samples = []
        ids = []
        
        # Generate synthetic samples (shifts/scales) for robustness
        # 1. Base
        samples.append(roi_base); ids.append(new_id)
        
        # 2. Flips
        samples.append(cv2.flip(roi_base, 1)); ids.append(new_id)
        
        # 3. Brightness/Contrast variations
        for alpha in [0.8, 1.2]:
            aug = cv2.convertScaleAbs(roi_base, alpha=alpha, beta=0)
            samples.append(aug); ids.append(new_id)

        try:
            # Check if model initialized
            self.recognizer.update(samples, np.array(ids))
            self.save()
            print(f"Registered {name} (ID: {new_id})")
            return True
        except:
            self.train(samples, ids)
            return True

    def delete_all_users(self):
        """Delete all trained data."""
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create() # Reset model
            self.labels = {}
            
            # Remove files
            if os.path.exists(self.model_path): os.remove(self.model_path)
            if os.path.exists(self.labels_path): os.remove(self.labels_path)
            
            print("All users deleted.")
            return True
        except Exception as e:
            print(f"Error deleting users: {e}")
            return False
