import os
import random

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen, ScreenManager
from plyer import tts
from torchvision import transforms
from PIL import Image as PILImage

# Load environment variables (must have GEMINI_API_KEY in your .env)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# -----------------------------------------------------------------------------
# Constants & Globals
# -----------------------------------------------------------------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
CLASS_NAMES = [
    "drowsy",
    "focused",
    "holding phone",
    "sleepy",
    "using phone",
    "yawning",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: str) -> torch.jit.ScriptModule:
    """Load and return a TorchScript model."""
    try:
        model = torch.jit.load(path, map_location=DEVICE)
        model.to(DEVICE).eval()
        print("Loaded TorchScript model:", path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at '{path}'")


def detect_face(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """
    Detect first face in the frame and return cropped face and coords.
    Returns (None, None) if no face detected.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )
    if len(faces) == 0:
        return None, None

    x, y, w, h = faces[0]
    return frame[y : y + h, x : x + w], (x, y, w, h)


def call_gemini(user_input: str) -> str:
    """Send a prompt to Gemini and return its reply."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(user_input),
            ],
        )
    ]
    config = types.GenerateContentConfig(
        temperature=0.7, top_p=0.9, top_k=40, max_output_tokens=150
    )

    response = ""
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash", contents=contents, config=config
    ):
        response += chunk.text
    return response or "No response from Gemini."


# -----------------------------------------------------------------------------
# Screen Definitions
# -----------------------------------------------------------------------------
class LoginScreen(Screen):
    def login(self):
        """Simple username==password check to move to menu."""
        if self.logname.text == self.password.text:
            self.manager.current = "menu"


class MenuScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.model = load_model("converted_model.pt")
        self.drowsy_count = 0
        self.focus_count = 0
        self.dist_count = 0
        self.llm_called = False

    def start_monitoring(self):
        """Begin video capture and schedule frame processing."""
        self.capture = cv2.VideoCapture("http://192.168.4.1:81/stream")
        Clock.schedule_interval(self.update_frame, 1 / 24)

        sys_msg = (
            "You are the Driver Monitoring Assistant. "
            "Detect drowsiness/distraction and give short advice."
        )
        self._send_to_gemini(sys_msg)

    def update_frame(self, dt):
        """Read a frame, detect face, classify state, update UI and alerts."""
        ret, frame = self.capture.read()
        if not ret:
            return

        face, coords = detect_face(frame)
        if face is not None:
            self._classify_and_annotate(face, frame, coords)
        else:
            self._handle_no_face()

        # display frame
        buf = cv2.flip(frame, 0).tobytes()
        tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
        tex.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
        self.image.texture = tex

    def _classify_and_annotate(self, face, frame, coords):
        """Run model inference, update counters, UI labels, and optionally alert."""
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = TRANSFORM(PILImage.fromarray(rgb)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
            conf, idx = torch.max(probs, 0)
            label = CLASS_NAMES[idx.item()]

        x, y, w, h = coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{label} ({conf:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            2,
        )

        getattr(self, f"_handle_{label.replace(' ', '_')}", self._reset_counters)()

        # update debug counters in UI
        self.count1.text = f"Focused: {self.focus_count}"
        self.count2.text = f"Drowsy: {self.drowsy_count}"
        self.count3.text = f"Distracted: {self.dist_count}"

    def _handle_focused(self):
        self.focus_count += 1
        if self.focus_count > 12:
            self.gpslabel.text = "FOCUSED"
            self._reset_counters(reset_llm=True)

    def _handle_drowsy(self):
        self.drowsy_count += 1
        if self.drowsy_count > 30:
            self.gpslabel.text = "SLEEPY"
            if not self.llm_called:
                self.llm_called = True
                self._send_to_gemini("Driver is sleepy, recommend a break.")

    def _handle_holding_phone(self):
        self._handle_distracted()

    def _handle_using_phone(self):
        self._handle_distracted()

    def _handle_yawning(self):
        self._handle_drowsy()

    def _handle_distracted(self):
        self.dist_count += 1
        if self.dist_count > 60 and not self.llm_called:
            self.gpslabel.text = "DISTRACTED"
            self.llm_called = True
            self._send_to_gemini("Driver is distracted, warn to focus on road.")

    def _reset_counters(self, reset_llm=False):
        self.drowsy_count = self.focus_count = self.dist_count = 0
        if reset_llm:
            self.llm_called = False

    def _send_to_gemini(self, text: str):
        """Send text to Gemini and display & speak the reply."""
        reply = call_gemini(text)
        self.chatout.text = reply
        tts.speak(reply)


class SettingsScreen(Screen):
    pass


class ChatScreen(Screen):
    pass


# -----------------------------------------------------------------------------
# App Entry Point
# -----------------------------------------------------------------------------
class DriverMonitoringApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(MenuScreen(name="menu"))
        sm.add_widget(SettingsScreen(name="settings"))
        sm.add_widget(ChatScreen(name="chat"))
        return sm


if __name__ == "__main__":
    DriverMonitoringApp().run()
