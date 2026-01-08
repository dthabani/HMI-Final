import cv2
import time
import math
import queue
import math
import queue
import threading
import numpy as np
import mediapipe as mp
from face_auth import FaceAuth
import speech_recognition as sr
from mediapipe.tasks import python
from gemini_parser import GeminiParser
from mediapipe.tasks.python import vision
from emotion_manager import EmotionManager
from PIL import Image, ImageDraw, ImageFont
from spotify_controller import SpotifyController

# Configuration
WIDTH, HEIGHT = 1280, 720
SMOOTHING = 5
VOL_MIN_DIST = 30
VOL_MAX_DIST = 170
UPDATE_INTERVAL = 0.1
EMOTION_CHECK_INTERVAL = 0.5 
EMOTION_BUFFER_SIZE = 10 

# Activation Timers (Seconds)
TIME_TO_ACTIVATE_PLAYLIST = 2.0  
TIME_TO_ACTIVATE_VOLUME = 2.0    
TIME_GESTURE_HOLD_RESET = 5.0 
AUTH_TIMEOUT = 120.0 # 2 Minutes Persistence

# UI Colors (BGR)
COLOR_ACTIVE = (0, 255, 127)   # Spring Green
COLOR_LOCKED = (50, 50, 50)    # Dark Gray
COLOR_TEXT = (255, 255, 255)   # White
COLOR_ACCENT = (255, 105, 180) # Hot Pink
COLOR_VOICE = (255, 255, 0)    # Cyan/Yellow
COLOR_GESTURE = (0, 200, 255)  # Orange/Gold
COLOR_DENIED = (0, 0, 255)     # Red
COLOR_LANDMARK = (0, 255, 0)   # Green for landmarks
COLOR_CONNECTION = (255, 255, 255) # White for connections

# Manual Landmark Drawing
# Hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks_manual(image, landmarks):
    """
    Draws hand landmarks and connections manually using OpenCV.
    Args:
    image: The BGR image to draw on.
    landmarks: A list of NormalizedLandmark objects.
    """
    h, w, c = image.shape
    
    # Pre-compute coordinates
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    # Draw Connections
    for connection in HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        cv2.line(image, coords[start_idx], coords[end_idx], COLOR_CONNECTION, 2)

    # Draw Landmarks
    for coord in coords:
        cv2.circle(image, coord, 5, COLOR_LANDMARK, cv2.FILLED)
        cv2.circle(image, coord, 5, COLOR_CONNECTION, 1) # Border

# Helper: Pillow Text Draw
def draw_text_pil(img, text, pos, size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("Arial.ttf", size)
    except IOError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except IOError:
            font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Threaded Spotify Worker
class SpotifyWorker(threading.Thread):
    def __init__(self):
        super().__init__()
        self.cmd_queue = queue.Queue()
        self.running = True
        self.sp = None
        self.connected = False
        self.display_name = "Connecting..."
        self.volume = 50 # Default start

    def run(self):
        try:
            self.sp = SpotifyController()
            self.display_name = self.sp.test_connection()
            self.connected = True
            print(f"Worker connected as {self.display_name}")
        except Exception as e:
            print(f"Worker Connect Error: {e}")
            self.display_name = "Connection Failed"

        while self.running:
            try:
                task = self.cmd_queue.get(timeout=0.1)
                cmd_type = task['type']
                
                if not self.connected: raise Exception("Not Connected")

                if cmd_type == 'volume':
                    vol = task['value']
                    print(f"Setting volume to {vol}%") # Feedback
                    self.sp.set_volume(vol)
                    self.volume = vol # Sync State
                    try: 
                        import subprocess
                        subprocess.Popen(["say", f"Setting volume to {vol} percent"])
                    except: pass
                elif cmd_type == 'search':
                    q = task['query']
                    # Feedback
                    # print(f"Searching Spotify for: {q}")
                    res = self.sp.search_and_play(q, task.get('search_type', 'playlist'), task.get('randomize', False))
                    if res: 
                        print(f"Playing: {res}")
                        # TTS Feedback (Async)
                        try:
                            import subprocess
                            # macOS 'say' command
                            text = f"Playing {res}"
                            subprocess.Popen(["say", text])
                        except Exception as e:
                            print(f"TTS Error: {e}")
                    else:
                        print("Spotify: No results found.")
                elif cmd_type == 'play_pause':
                    print("Toggling Play/Pause")
                    self.sp.play_pause_track()
                    try: 
                        import subprocess
                        subprocess.Popen(["say", "Toggling playback"])
                    except: pass
                elif cmd_type == 'next':
                    print("Skipping to Next Track")
                    self.sp.next_track()
                    try: 
                        import subprocess
                        subprocess.Popen(["say", "Playing next track"])
                    except: pass
                elif cmd_type == 'prev':
                    print("Going to Previous Track")
                    self.sp.previous_track()
                    try: 
                        import subprocess
                        subprocess.Popen(["say", "Playing previous track"])
                    except: pass
                elif cmd_type == 'shuffle':
                    print(f"Setting Shuffle: {task['state']}")
                    self.sp.toggle_shuffle(task['state'])
                    try: 
                        import subprocess
                        status = "on" if task['state'] else "off"
                        subprocess.Popen(["say", f"Shuffle is now {status}"])
                    except: pass
                elif cmd_type == 'transfer':
                    print(f"Transferring Playback to: {task['device']}")
                    self.sp.transfer_playback(task['device'])
                    try: 
                        import subprocess
                        subprocess.Popen(["say", f"Transferring playback to {task['device']}"])
                    except: pass
                elif cmd_type == 'queue':
                    print(f"Adding URI to Queue: {task['uri']}")
                    self.sp.add_to_queue(task['uri'])
                    try: 
                        import subprocess
                        subprocess.Popen(["say", "Added to queue"])
                    except: pass
                elif cmd_type == 'seek':
                    print(f"Seeking to: {task['position']}ms")
                    self.sp.seek_track(task['position'])
                    try: 
                        import subprocess
                        # Convert ms back to seconds for speech
                        sec = int(task['position'] / 1000)
                        subprocess.Popen(["say", f"Seeking to {sec} seconds"])
                    except: pass
                elif cmd_type == 'search_queue':
                    q = task['query']
                    print(f"Searching to Queue: {q}")
                    self.sp.search_and_queue(q)

            except queue.Empty: continue
            except Exception as e: print(f"Worker Error: {e}")

    def set_volume(self, vol): self.cmd_queue.put({'type': 'volume', 'value': vol})
    def search_playlist(self, query): self.cmd_queue.put({'type': 'search', 'query': query, 'search_type': 'playlist', 'randomize': True})
    def search_track(self, query): self.cmd_queue.put({'type': 'search', 'query': query, 'search_type': 'track', 'randomize': False})
    def play_pause(self): self.cmd_queue.put({'type': 'play_pause'})
    def next_track(self): self.cmd_queue.put({'type': 'next'})
    def prev_track(self): self.cmd_queue.put({'type': 'prev'})
    
    # New Advanced Controls
    def toggle_shuffle(self, state): self.cmd_queue.put({'type': 'shuffle', 'state': state})
    def transfer_playback(self, device): self.cmd_queue.put({'type': 'transfer', 'device': device})
    def add_to_queue(self, uri): self.cmd_queue.put({'type': 'queue', 'uri': uri})
    def seek_track(self, ms): self.cmd_queue.put({'type': 'seek', 'position': ms})
    def search_queue(self, query): self.cmd_queue.put({'type': 'search_queue', 'query': query})
    
    def stop(self): self.running = False

# Voice Thread
class VoiceThread(threading.Thread):
    def __init__(self, spotify_worker):
        super().__init__()
        self.spotify_worker = spotify_worker
        self.running = True
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False 
        self.status = "Voice: OFF"
        self.access_granted = False # Controlled by Main
        self.needs_calibration = True

    def toggle_listening(self):
        if not self.access_granted: return False
        self.listening = not self.listening
        
        # Terminal Feedback
        if self.listening:
            print("Voice control ENABLED")
            self.status = "Voice: LISTENING..."
            self.needs_calibration = True # Recalibrate when enabling
        else:
            print("Voice control DISABLED")
            self.status = "Voice: OFF"
            
        return self.listening
    
    def set_access(self, granted):
        self.access_granted = granted
        if not granted: 
            self.listening = False
            self.status = "Voice: LOCKED"

    def run(self):
        # Initialize Gemini Parser

        try:
            parser = GeminiParser()
        except:
            parser = None
            print("Gemini Parser Init Failed - Fallback to regex")

        while self.running:
            if self.listening and self.access_granted:
                try:
                    with self.microphone as source:
                        if self.needs_calibration:
                            print("Calibrating microphone for ambient noise...")
                            self.status = "Calibrating..."
                            self.recognizer.adjust_for_ambient_noise(source, duration=1)
                            self.needs_calibration = False
                            print("Microphone Calibrated.")
                            self.status = "Voice: LISTENING..."

                        print("Listening for voice command...") # Terminal
                        self.status = "Listening..." # UI
                        # Increase phrase_time_limit to allow longer commands
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=8)
                    
                    self.status = "Processing..."
                    command = self.recognizer.recognize_google(audio).lower()
                    
                    # Feedback: What was heard
                    print(f"Voice heard: {command}")
                    
                    # Auto-Disable Voice
                    self.listening = False
                    print("Voice control DISABLED") 
                    self.status = f"Heard: {command}" 
                    
                    # GEMINI INTEGRATION
                    parsed = None
                    if parser:
                        self.status = "Voice: Asking Gemini..."
                        parsed = parser.parse_command(command)
                    
                    if parsed and parsed.get('intent'):
                        intent = parsed['intent']
                        # Feedback: Gemini Intent
                        print(f"Gemini Intent: {intent}")
                        self.status = f"Intent: {intent}"
                        
                        if intent == "PLAY_SONG":
                            q = parsed.get('search_query') or f"{parsed.get('song')} {parsed.get('artist')}"
                            # Feedback: Searching
                            print(f"Searching Spotify for: Track '{q}'")
                            self.status = f"Searching: {q}"
                            self.spotify_worker.search_track(q)
                            
                        elif intent == "PLAY_PLAYLIST" or intent == "PLAY_GENRE_OR_VIBE":
                            q = parsed.get('search_query') or parsed.get('playlist')
                            # Feedback: Searching
                            print(f"Searching Spotify for: Playlist '{q}'")
                            self.status = f"Searching: {q}"
                            self.spotify_worker.search_playlist(q)

                        elif intent == "PLAY_ARTIST":
                            artist = parsed.get('artist') or parsed.get('search_query')
                            # Feedback: Searching
                            print(f"Searching Spotify for: Artist '{artist}'")
                            self.status = f"Searching: {artist}"
                            self.spotify_worker.search_track(f"artist:{artist}") 

                        elif intent == "PLAY_RADIO":
                            q = parsed.get('search_query')
                            # Feedback: Searching
                            print(f"Searching Spotify for: Radio '{q}'")
                            self.status = f"Searching: {q}"
                            self.spotify_worker.search_playlist(f"{q}")
                            
                        elif intent == "PAUSE" or intent == "RESUME":
                            print("Executing: Play/Pause")
                            self.status = "Exec: Play/Pause"
                            self.spotify_worker.play_pause()
                        
                        elif intent == "NEXT_TRACK":
                            print("Executing: Next Track")
                            self.status = "Exec: Next Track"
                            self.spotify_worker.next_track()
                        
                        elif intent == "PREVIOUS_TRACK":
                            print("Executing: Previous Track")
                            self.status = "Exec: Prev Track"
                            self.spotify_worker.prev_track()
                        
                        elif intent == "SET_VOLUME":
                            vol = parsed.get('volume')
                            if vol is not None: 
                                print(f"Executing: Set Volume to {vol}%")
                                self.status = f"Vol: {vol}%"
                                self.spotify_worker.set_volume(vol)
                        
                        elif intent == "SHUFFLE_ON":
                            print("Executing: Shuffle ON")
                            self.status = "Shuffle: ON"
                            self.spotify_worker.toggle_shuffle(True)
                            
                        elif intent == "SHUFFLE_OFF":
                            print("Executing: Shuffle OFF")
                            self.status = "Shuffle: OFF"
                            self.spotify_worker.toggle_shuffle(False)
                        
                        elif intent == "TRANSFER_PLAYBACK":
                            dev = parsed.get('device_name')
                            if dev:
                                print(f"Executing: Transfer to '{dev}'")
                                self.status = f"Transfer: {dev}"
                                self.spotify_worker.transfer_playback(dev)

                        elif intent == "ADD_TO_QUEUE":
                            # We need to search for the URI first
                            q = parsed.get('search_query') or f"{parsed.get('song')} {parsed.get('artist')}"
                            if q:
                                print(f"Queueing: Finding '{q}'...")
                                # Note: This requires a helper or creating a 'search_and_queue' in worker
                                # Let's implement a 'search_queue' in worker.
                                self.spotify_worker.cmd_queue.put({'type': 'search_queue', 'query': q})
                                self.status = f"Queueing: {q}"

                        elif intent == "SEEK_TO_POSITION":
                            secs = parsed.get('timestamp_seconds')
                            if secs is not None:
                                ms = int(secs * 1000)
                                print(f"Executing: Jump to {secs}s")
                                self.status = f"Seek: {secs}s"
                                self.spotify_worker.seek_track(ms)
                            
                        else:
                            print(f"Unknown Intent: {intent}")
                            self.status = "Voice: Unknown Intent"

                    else:
                        # FALLBACK LOGIC (Deterministic)
                        
                        # Advanced playback controls that require Gemini
                        ADVANCED_KEYWORDS = [
                            "shuffle", "queue", "add to", "transfer", "play on", 
                            "jump to", "seek", "skip", "next", "previous", "back", 
                            "pause", "resume", "stop", "volume", "turn up", "turn down"
                            ]

                        if any(keyword in command for keyword in ADVANCED_KEYWORDS):
                            msg = f"Cannot execute {command}. Advanced controls require Gemini connection."
                            print(f"Voice: {msg}")
                            self.status = "Voice: CMD Not Supported"
                            # TTS Feedback
                            try:
                                import subprocess
                                subprocess.Popen(["say", msg])
                            except: pass

                        # Playlist Search
                        elif "playlist" in command:
                            query = command.split("playlist", 1)[1].strip()
                            if query:
                                print(f"Fallback: Searching Playlist '{query}'")
                                self.status = f"FB: Playlist {query}"
                                self.spotify_worker.search_playlist(query)

                        # Song Search
                        elif "song" in command:
                            query = command.split("song", 1)[1].strip()
                            if query:
                                print(f"Fallback: Searching Track '{query}'")
                                self.status = f"FB: Track {query}"
                                self.spotify_worker.search_track(query)

                        # Default Play
                        elif "play" in command:
                            query = command.split("play", 1)[1].strip()
                            if query:
                                print(f"Fallback: Searching Track '{query}'")
                                self.status = f"FB: Track {query}"
                                self.spotify_worker.search_track(query)
                    
                    time.sleep(1)
                    if self.listening: self.status = "Voice: LISTENING..."
                except sr.WaitTimeoutError:
                    print("Voice: Timeout (No speech detected).")
                    if self.listening: self.status = "Voice: ..."
                except sr.UnknownValueError:
                    print("Voice: Could not understand audio.")
                    if self.listening: self.status = "Voice: ?"
                except Exception as e:
                    print(f"Voice Error: {e}")
            else: time.sleep(0.5)
    def stop(self): self.running = False

# Emotion Thread Logic moved to emotion_manager.py

def main():
    # System Initialization
    print("Initializing System...")
    spotify_worker = SpotifyWorker(); spotify_worker.start()
    emotion_manager = EmotionManager()
    voice_thread = VoiceThread(spotify_worker); voice_thread.start()
    
    face_auth = FaceAuth() # Load model

    # MediaPipe Setup
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.VIDEO,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.7,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.5)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH); cap.set(4, HEIGHT)

    # Long-Term State (Persists across frames)
    # Application Modes
    MODE_MAIN = 0
    MODE_MENU = 1
    MODE_REG_COUNTDOWN = 2
    MODE_REG_INPUT = 3
    
    app_mode = MODE_MAIN
    
    # Timers & Counters
    right_hand_timer = 0
    
    last_action_time = 0
    last_auth_time = 0
    last_voice_toggle_time = 0
    last_gesture_time = 0
    last_playlist_search_time = 0
    last_vol_change_time = 0
    
    # FPS State
    fps_display = 0
    latency_display = 0
    last_fps_update_time = 0

    # Registration State
    reg_start_time = 0
    reg_countdown_duration = 5
    reg_img_capture = None
    input_name_buffer = ""

    # Flags & Globals
    COOLDOWN_TIME = 5.0
    AUTH_TIMEOUT = 120.0
    
    is_busy = False
    access_granted = False
    current_user = "Unknown"
    gesture_hold_active = False
    current_vol = 50 
    
    p_time = 0
    
    print("System Ready. Press 'q' to quit. Press 'r' to Open Menu.")

    while True:
        # Input Phase
        success, img = cap.read()
        if not success: break
        
        img = cv2.flip(img, 1)
        current_time = time.time()
        
        # Frame State Initialization
        found_right = False
        volume_update_val = None
        system_status = "Idle"
        results = None
        hand_landmarks_list = []
        handedness_list = []

        # FPS Smoothing
        latency_ms_inst = int((current_time - p_time) * 1000)
        if current_time - last_fps_update_time > 0.5:
            # Calculate momentary FPS
            curr = 1 / (current_time - p_time) if (current_time - p_time) > 0 else 0
            fps_display = int(curr)
            latency_display = latency_ms_inst # Smooth update
            last_fps_update_time = current_time
        p_time = current_time

        # State Machine Logic
        
        # MODE: MAIN (Normal Operation)
        if app_mode == MODE_MAIN:
            
            # Global Updates (Auth, Busy)
            if is_busy and (current_time - last_action_time > 15.0):
                is_busy = False
                print("System auto-unlocked (timeout)")

            if (current_time - last_auth_time < AUTH_TIMEOUT) and (current_user != "Unknown"):
                access_granted = True
            else:
                if access_granted:
                    print("Authentication expired.")
                    access_granted = False
                
                # Predict Face (Non-blocking)
                user, conf = face_auth.predict(img)
                if user != "Unknown" and user != "No Face":
                    if current_user != user: print(f"User authenticated: {user}")
                    current_user = user
                    access_granted = True
                    last_auth_time = current_time
                else:
                    current_user = "Unknown"
                    access_granted = False
            
            voice_thread.set_access(access_granted and not is_busy)
            
            # Status Text Logic
            if not access_granted: system_status = "Locked"
            elif is_busy: system_status = "Busy..."
            elif voice_thread.listening: system_status = "Listening"
            
            # Detection
            if access_granted:

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                detection_result = detector.detect_for_video(mp_image, int(current_time * 1000))
                if detection_result.hand_landmarks:
                    results = detection_result
                    hand_landmarks_list = results.hand_landmarks
                    handedness_list = results.handedness

            if access_granted and results and not is_busy and (current_time - last_action_time > COOLDOWN_TIME):
                # Prepare Hand Labels
                hand_labels = []
                for i in range(len(handedness_list)):
                    org = handedness_list[i][0].category_name
                    hand_labels.append("Left" if org == "Right" else "Right")
                
                for idx, hand_lms in enumerate(hand_landmarks_list):
                    label = hand_labels[idx]
                    
                    # IGNORE LEFT HAND
                    if label == "Left":
                        continue

                    h, w, c = img.shape
                    lm_list = []
                    for id, lm in enumerate(hand_lms):
                        lm_list.append([id, int(lm.x * w), int(lm.y * h)])
                    
                    if len(lm_list) >= 21:
                        wrist_x, wrist_y = lm_list[0][1], lm_list[0][2]
                        # Fingers Open
                        fingers_indices = [8, 12, 16, 20]; knuckles_indices = [6, 10, 14, 18]
                        fingers_open = []
                        for i in range(4):
                            tx, ty = lm_list[fingers_indices[i]][1], lm_list[fingers_indices[i]][2]
                            kx, ky = lm_list[knuckles_indices[i]][1], lm_list[knuckles_indices[i]][2]
                            dt = math.hypot(tx - wrist_x, ty - wrist_y)
                            dk = math.hypot(kx - wrist_x, ky - wrist_y)
                            fingers_open.append(dt > dk * 1.3)
                        
                        tx, ty = lm_list[4][1], lm_list[4][2]
                        ipx, ipy = lm_list[3][1], lm_list[3][2]
                        thumb_open = math.hypot(tx - wrist_x, ty - wrist_y) > math.hypot(ipx - wrist_x, ipy - wrist_y) * 1.1
                        
                        # REMOVED: is_palm_open logic (Playback)
                        # REMOVED: Pinch logic (Volume)

                        is_peace = fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]
                        
                        # Triggers
                        # Trigger: Thumb Open AND Fingers Closed (Thumbs Up)
                        is_thumbs_up = thumb_open and not fingers_open[1] and not fingers_open[2] and not fingers_open[3]
                        
                        if is_thumbs_up:
                            # Debounce/Hold logic
                            right_hand_timer += 1
                            found_right = True # Visual feedback
                            
                            if right_hand_timer > 30: # Hold for ~1 sec
                                is_busy = True
                                print("Analyzing Mood...")
                                system_status = "Reading Mood..."
                                
                                # Show feedback before freezing for inference
                                cv2.putText(img, "Reading Mood...", (WIDTH//2 - 100, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                                cv2.imshow("Spotify Gesture Control", img)
                                cv2.waitKey(1)
                                
                                # ONE-SHOT INFERENCE
                                detected_mood = emotion_manager.detect_emotion(img)
                                
                                if detected_mood:
                                    print(f"Mood Detected: {detected_mood}")
                                    system_status = f"Mood: {detected_mood}"
                                    
                                    # Auto-Disable Mood Search
                                    print("Mood search DISABLED")
                                    is_thumbs_up = False 
                                    
                                    # Send to Gemini for creative search query
                                    smart_query = ""
                                    try:
                                        parser = GeminiParser()
                                        gemini_result = parser.generate_search_query_from_mood(detected_mood)
                                        
                                        if gemini_result:
                                            smart_query = gemini_result
                                            print(f"Gemini suggests: '{smart_query}' for mood '{detected_mood}'")
                                        else:
                                            # Clean Fallback
                                            print("INFO:GeminiParser:Gemini failed or returned null. Using fallback logic.")
                                            smart_query = f"{detected_mood} mood songs" # Default fallback
                                            
                                    except Exception:
                                        print("INFO:GeminiParser:Gemini failed or returned null. Using fallback logic.")
                                        smart_query = f"{detected_mood} mood songs" # Default fallback

                                    # Send to Spotify
                                    spotify_worker.search_playlist(smart_query)
                                else:
                                    print("Mood detection failed.")
                                    system_status = "Mood Failed"
                                
                                last_action_time = current_time
                                is_busy = False
                                right_hand_timer = 0
                        
                        elif is_peace:
                            if current_time - last_voice_toggle_time > 2.0:
                                voice_thread.toggle_listening()
                                last_voice_toggle_time = current_time
                        
                        if not is_thumbs_up: right_hand_timer = 0
                        # REMOVED: Left hand timer logic

            # Render Main
            # Landmarks
            if results:
                for hand_lms in hand_landmarks_list: draw_landmarks_manual(img, hand_lms)
            
            # Status Bar
            cv2.rectangle(img, (0, 0), (WIDTH, 50), (20, 20, 20), cv2.FILLED)
            
            # Status
            status_c = COLOR_ACTIVE if access_granted else COLOR_DENIED
            if is_busy: status_c = (255, 255, 0)
            img = draw_text_pil(img, f"Status: {system_status}", (20, 15), size=20, color=status_c)

            # Right Side Info Group
            # Latency
            img = draw_text_pil(img, f"Lat: {latency_display}ms", (WIDTH - 250, 15), size=18, color=COLOR_TEXT)
            
            # FPS
            img = draw_text_pil(img, f"FPS: {fps_display}", (WIDTH - 120, 15), size=18, color=COLOR_TEXT)

            if access_granted:
                # User
                img = draw_text_pil(img, f"User: {current_user}", (WIDTH - 550, 15), size=18, color=COLOR_GENERIC_TEXT if 'COLOR_GENERIC_TEXT' in globals() else COLOR_TEXT)
                # Mood
                mood_text = "Mood: Idle"
                mood_color = (128, 0, 128) # Purple (Idle)
                
                if system_status == "Reading Mood...":
                    mood_text = "Mood: DETECTING..."
                    mood_color = (0, 255, 255) # Yellow
                elif "Mood:" in system_status: 
                    mood_text = system_status 
                    mood_color = (255, 0, 255) # Magenta/Purple (Active)
                
                img = draw_text_pil(img, mood_text, (WIDTH - 400, 15), size=18, color=mood_color)
            
            # Volume Bar 
            # Sync volume from Worker
            current_vol = spotify_worker.volume

            bar_x = 30; bar_y_start=150; bar_y_end=500
            vol_bar_h = np.interp(current_vol, [0, 100], [bar_y_end, bar_y_start])
            c_bar = COLOR_LOCKED # Default color since pinch is removed
            cv2.rectangle(img, (bar_x, bar_y_start), (bar_x + 20, bar_y_end), (40, 40, 40), 2)
            cv2.rectangle(img, (bar_x, int(vol_bar_h)), (bar_x + 20, bar_y_end), c_bar, cv2.FILLED)
            img = draw_text_pil(img, f'{int(current_vol)}%', (bar_x - 5, bar_y_end + 10), size=16, color=c_bar)
            
            # Visual Feedback
            # REMOVED: Left Hand Feedback
            if found_right:
                col = COLOR_ACTIVE # Always active if found
                cv2.circle(img, (WIDTH-60, HEIGHT-60), 30, col, -1)
                img = draw_text_pil(img, "R", (WIDTH-68, HEIGHT-72), size=24, color=(0,0,0))

        # MODE: MENU (Selection)
        elif app_mode == MODE_MENU:
            # Darken Background
            overlay = img.copy()
            cv2.rectangle(overlay, (0,0), (WIDTH, HEIGHT), (0,0,0), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Menu Box
            cx, cy = WIDTH//2, HEIGHT//2
            cv2.rectangle(img, (cx-300, cy-200), (cx+300, cy+200), (30, 30, 30), cv2.FILLED)
            cv2.rectangle(img, (cx-300, cy-200), (cx+300, cy+200), (255, 255, 255), 2)
            
            img = draw_text_pil(img, "ADMIN MENU", (cx-100, cy-180), size=30, color=COLOR_ACTIVE)
            img = draw_text_pil(img, "[1] Add New User", (cx-200, cy-80), size=24, color=COLOR_TEXT)
            img = draw_text_pil(img, "[2] Delete All Users", (cx-200, cy), size=24, color=COLOR_DENIED)
            img = draw_text_pil(img, "[Q] Return to App", (cx-200, cy+100), size=24, color=(200,200,200))

        # MODE: COUNTDOWN (Registration)
        elif app_mode == MODE_REG_COUNTDOWN:
            elapsed = current_time - reg_start_time
            remaining = int(reg_countdown_duration - elapsed)
            
            # Draw Frame
            img = draw_text_pil(img, f"CAPTURING IN {remaining+1}...", (WIDTH//2 - 150, HEIGHT//2), size=50, color=COLOR_ACCENT)
            
            if remaining < 0:
                # Capture!
                reg_img_capture = img.copy()
                app_mode = MODE_REG_INPUT
                input_name_buffer = ""

        # MODE: INPUT (Name Entry)
        elif app_mode == MODE_REG_INPUT:
            # Freeze frame background
            if reg_img_capture is not None: img = reg_img_capture.copy()
            
            # Input Box
            cx, cy = WIDTH//2, HEIGHT//2
            cv2.rectangle(img, (cx-300, cy-100), (cx+300, cy+100), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(img, (cx-300, cy-100), (cx+300, cy+100), (255, 255, 255), 2)
            
            img = draw_text_pil(img, "ENTER NAME:", (cx-280, cy-80), size=24, color=COLOR_ACTIVE)
            img = draw_text_pil(img, input_name_buffer + "|", (cx-280, cy), size=30, color=COLOR_TEXT)
            img = draw_text_pil(img, "Press ENTER to Save", (cx-280, cy+60), size=18, color=(150,150,150))

        # DISPLAY
        cv2.imshow("Spotify Gesture Control", img)
        
        # INPUT HANDLING
        key = cv2.waitKey(1) & 0xFF
        if app_mode == MODE_MAIN:
            if key == ord('q'): break
            elif key == ord('r'): app_mode = MODE_MENU
        
        elif app_mode == MODE_MENU:
            if key == ord('q'): app_mode = MODE_MAIN
            elif key == ord('1'): # Add
                app_mode = MODE_REG_COUNTDOWN
                reg_start_time = time.time()
            elif key == ord('2'): # Delete
                face_auth.delete_all_users()
                print("All users deleted. System Locked.")
                # Security Lockout
                current_user = "Unknown"
                access_granted = False
                last_auth_time = 0
                system_status = "Locked"
                app_mode = MODE_MAIN
        
        elif app_mode == MODE_REG_COUNTDOWN:
            if key == ord('q'): app_mode = MODE_MAIN # Cancel
        
        elif app_mode == MODE_REG_INPUT:
            if key == 13: # Enter
                if len(input_name_buffer) > 0:
                    print(f"Registering: {input_name_buffer}")
                    # Ensure there is a valid capture
                    if reg_img_capture is not None:
                        face_auth.register_new_user(reg_img_capture, input_name_buffer)
                        face_auth.load()
                    app_mode = MODE_MAIN
            elif key == 8 or key == 127: # Backspace
                input_name_buffer = input_name_buffer[:-1]
            elif 32 <= key <= 126: # Ascii
                input_name_buffer += chr(key)
            elif key == ord('q'): app_mode = MODE_MAIN # Cancel

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    spotify_worker.stop()
    # emotion_thread.stop()
    voice_thread.stop()
    print("System Shutdown.")

if __name__ == "__main__":
    main()
