import cv2
import torch
import timm
import numpy as np
import requests
from torchvision import transforms
from PIL import Image as im
import random

import kivy
import kivymd
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivymd.uix.textfield import textfield


AWB = True

# Load OpenCV's built-in face detector  ,# Example: 6 classes for driver monitoring
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
num_classes = 6; class_names = ['DangerousDriving', 'Distracted', 'Drinking', 'SafeDriving', 'SleepyDriving', 'Yawn']

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('efficientnet_lite0', pretrained=False)
model.reset_classifier(num_classes)
model.load_state_dict(torch.load("best_efficientnet_lite_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure size matches training data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
    ])

def simulate_ir_effect(frame):
    """Converts an RGB frame to a simulated infrared-like grayscale image.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)  # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def detect_and_crop_face(frame):
    """Detects a face in the frame and crops it.
    Also returns face coordinates (x, y, w, h) for drawing a bounding box.  If no face is found, returns None."""
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face_crop = frame[y:y+h, x:x+w]  # Crop face region
        return face_crop, (x, y, w, h)
    return None, None



class Myroot(BoxLayout):
    def __init__(self):
        super(Myroot, self).__init__()
        self.capture = cv2.VideoCapture()
        
    
    def start(self): 
        URL  = self.inp.text
        self.label.text = URL
        self.capture = cv2.VideoCapture("http://192.168.4.1:81/stream")
        Clock.schedule_interval(self.loadVid , 1.0/24.0)

    def loadVid(self,*args):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
        ret, frame = self.capture.read()

        ir_like_frame = simulate_ir_effect(frame)
        face, face_coords = detect_and_crop_face(ir_like_frame)
        
        if face is not None: # Transform for model input
            rgb_face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
            pil_image = im.fromarray(rgb_face)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Convert logits to probabilities
                confidence, predicted = torch.max(probabilities, 0)  # Get highest confidence prediction
                predicted_label = class_names[predicted.item()]
                print(predicted_label)
                self.testlabel.text =predicted_label


            x, y, w, h = face_coords
            cv2.rectangle(ir_like_frame, (x, y), (x+w, y+h), (255, 255, 255), 2) 
            cv2.putText(ir_like_frame, f'{predicted_label} ({confidence:.2f})', 
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

   
    
        self.image_frame = ir_like_frame
        buffer= cv2.flip(ir_like_frame, 0).tostring()
        texture = Texture.create(size = (frame.shape[1], frame.shape[0]),colorfmt = 'luminance')
        texture.blit_buffer(buffer, colorfmt = 'luminance',bufferfmt = 'ubyte')
        self.image.texture = texture
        




    
    

class aTester(App):
    def build(self):
        return Myroot()
    

tester = aTester()
tester.run()