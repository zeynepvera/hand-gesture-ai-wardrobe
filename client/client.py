import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import mediapipe as mp
import time
import requests
from core.feature_extractor import extract_features, get_combination_vector
from services.weather import get_my_city

SERVER_URL = "http://127.0.0.1:8000"

MY_CITY = get_my_city()


def api_save_combination(top, bottom, combo_vector):
    """
    Sends a POST request to /save_combination endpoint.
    Asks the server to save a new combination to the database.
    """
    try:
        response = requests.post(f"{SERVER_URL}/save_combination", json={
            "top":          top,
            "bottom":       bottom,
            "combo_vector": combo_vector
        })
        return response.json()
    except Exception as e:
        print(f"Server error (save): {e}")
        return None
 
 
def api_like(top, bottom):
    """
    Sends a POST request to /like endpoint.
    Asks the server to:
      1. Update liked=1 in database
      2. Retrain the ML model
    """
    try:
        response = requests.post(f"{SERVER_URL}/like", json={
            "top":    top,
            "bottom": bottom
        })
        return response.json()
    except Exception as e:
        print(f"Server error (like): {e}")
        return None
 
 
def api_predict(combo_vector):
    """
    Sends a POST request to /predict endpoint.
    Asks the server what the match % is for this combo vector.
    Returns a number between 0.0 and 1.0
    """
    try:
        response = requests.post(f"{SERVER_URL}/predict", json={
            "combo_vector": combo_vector
        })
        data = response.json()
        return data.get("match_score", 0.5)
    except Exception as e:
        print(f"Server error (predict): {e}")
        return 0.5  # return neutral 50% if server is unreachable
 
 
def api_status():
    """
    Sends a GET request to /status endpoint.
    Returns total and liked count from database.
    """
    try:
        response = requests.get(f"{SERVER_URL}/status")
        return response.json()
    except Exception as e:
        print(f"Server error (status): {e}")
        return {"total": 0, "liked": 0, "model_ready": False}

def api_weather(city):
    """
    Asks the server for current weather data for the given city.
    Returns weather dict or None if server is unreachable.
    """
    try:
        response = requests.get(f"{SERVER_URL}/weather", params={"city": city})
        return response.json()
    except Exception as e:
        print(f"Server error (weather): {e}")
        return None


# MEDIAPIE SETUP
mp_hands = mp.solutions.hands
mp_draw= mp.solutions.drawing_utils

hands= mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7


)

# SETTINGS 
ASSETS_FOLDER = "assets"

# Top outfit position and size
TOP_X, TOP_Y  = 170, 30
TOP_WIDTH     = 250
TOP_HEIGHT    = 220

# Bottom outfit position and size
BOT_X, BOT_Y  = 170, 245
BOT_WIDTH     = 250
BOT_HEIGHT    = 220

SWIPE_HISTORY_SIZE= 8
SWIPE_SPEED_THRESHOLD= 800
SWIPE_COOLDOWN= 0.6

THUMBS_UP_COOLDOWN= 1.5
THUMBS_UP_MARGIN= 30

WEATHER_REFRESH_INTERVAL = 600

#OUTFITS LOADER

def load_outfits(folder, prefix_has_underscore):
    
    files = []
    for f in os.listdir(folder):
        if not f.endswith(".png"):
            continue
        # Check naming pattern
        name = f.replace(".png", "")  #  "outfit_1" or "outfit1"
        if prefix_has_underscore and "_" in name:
            files.append(f)
        elif not prefix_has_underscore and "_" not in name:
            files.append(f)
    files.sort()
    return files


def load_images(files, folder, width, height):
    """
    Reads image files and resizes them.
    Returns a list of images with alpha channel (BGRA).
    """
    images = []
    for f in files:
        path = os.path.join(folder, f)
        img  = cv2.imread(path, cv2.IMREAD_UNCHANGED) #if i write only path the backrgound could be with, so we protect alpha channeş as well.
        if img is None:
            print(f"Warning: Could not load {f}")
            continue
        img = cv2.resize(img, (width, height))
        images.append(img)
    return images


def overlay_png(background, overlay, x, y):
    """
    Places a transparent PNG onto the background frame at position (x, y).
    Uses the alpha channel for blending.
    """
    h, w = overlay.shape[:2]

    # Make sure overlay fits within the frame
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    if overlay.shape[2] == 4:  # PNG has alpha channel
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = overlay[:, :, :3]

    return background

#swipe detection
def detect_swipe(history):
    
    if len(history) < 2:
        return None
    
    x_old, y_old, t_old = history[0]
    x_new, y_new, t_new = history[-1]
    time_passed= t_new- t_old

    if time_passed == 0:
        return None
    
    x_diff= x_new-x_old
    speed= abs(x_diff)/ time_passed

    if speed > SWIPE_SPEED_THRESHOLD:
        if x_diff>0:
            return "right"
        else:
            return "left"
        
    return None    

def detect_thumbs_up(hand_landmarks, frame_height):

    thumb_y= hand_landmarks.landmark[4].y * frame_height #L4 thumb tip
    index_y = hand_landmarks.landmark[8].y * frame_height
    middle_y= hand_landmarks.landmark[12].y * frame_height
    ring_y   = hand_landmarks.landmark[16].y * frame_height
    pinky_y  = hand_landmarks.landmark[20].y * frame_height

    highest_other_finger_y= min(index_y, middle_y, ring_y, pinky_y)

    return thumb_y < (highest_other_finger_y - THUMBS_UP_MARGIN)


#  LOAD OUTFITS
top_files = load_outfits(ASSETS_FOLDER, prefix_has_underscore=False)
bot_files = load_outfits(ASSETS_FOLDER, prefix_has_underscore=True)
top_images = load_images(top_files, ASSETS_FOLDER, TOP_WIDTH, TOP_HEIGHT)
bot_images = load_images(bot_files, ASSETS_FOLDER, BOT_WIDTH, BOT_HEIGHT)

top_index = 0 
bot_index = 0   

#extracting features once from all outfits
print("extracting features from outfits")
top_features_dict = {}
bot_features_dict = {}

for filename in top_files:
    path     = os.path.join(ASSETS_FOLDER, filename)
    image    = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    features = extract_features(image)
    top_features_dict[filename] = features

for filename in bot_files:
    path     = os.path.join(ASSETS_FOLDER, filename)
    image    = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    features = extract_features(image)
    bot_features_dict[filename] = features

first_top= top_files[0]
first_bot = bot_files[0]
first_vector = get_combination_vector(

    top_features_dict[first_top],
    bot_features_dict[first_bot]
)

# Check if server is reachable before starting
print("Connecting to server...")
status = api_status()
if status["total"] == 0 and status["liked"] == 0:
    print("Warning: Could not reach server. Make sure server is running!")
    print("Run in a separate terminal: uvicorn server:app --reload\n")
else:
    print(f"Server connected! DB: {status['total']} combinations, {status['liked']} liked")
    print(f"Model ready: {status['model_ready']}\n")
 
# Save first combination through server
api_save_combination(first_top, first_bot, first_vector)

# Fetch weather once at startup
print(f"Fetching weather for {MY_CITY}...")
weather_data         = api_weather(MY_CITY)
last_weather_time    = time.time()
 
# Build the weather text to show on screen
# OpenCV doesn't support emojis so we use text instead of emoji
if weather_data and weather_data.get("success"):
    weather_text  = f"{weather_data['city']} {weather_data['temp']}C {weather_data['condition']}"
    print(f"Weather: {weather_text}")
else:
    weather_text  = "Weather: unavailable"
    print("Could not fetch weather.")
 

# swipe state variables
finger_history= []
last_swipe_time= 0
swipe_feedback_text = ""
swipe_feedback_timer =0

#thumbs up state variable
last_thumbs_up_time= 0
thumbs_up_feedback_text= ""
thumbs_up_feedback_timer= 0

liked_combinations= []

# Get initial DB count from server
status   = api_status()
db_total = status["total"]
db_liked = status["liked"]

# CAMERA
camera = cv2.VideoCapture(0) #default webcam

if not camera.isOpened():
    print("Error: Camera could not be opened!")
    exit()

print("Camera opened. Press 'q' to quit.")
print("Swipe top half    --> change top outfit")
print("Swipe bottom half --> change bottom outfit")
print("Thumbs up         --> like current combination")

while True:
    success, frame = camera.read()
    if not success:
        print("Error: Frame could not be read!")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape #get frame size forzone check

    #mediapipe wants rgb but opencv gives bgr
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_top_vec   = top_features_dict[top_files[top_index]]
    current_bot_vec   = bot_features_dict[bot_files[bot_index]]
    current_combo_vec = get_combination_vector(current_top_vec, current_bot_vec)

    # Ask server for prediction instead of calling ML model directly
    match_score = api_predict(current_combo_vec)
  
    # Refresh weather every WEATHER_REFRESH_INTERVAL seconds
    # We don't refresh every frame — only every 10 minutes
    now_time = time.time()
    if now_time - last_weather_time > WEATHER_REFRESH_INTERVAL:
        weather_data      = api_weather(MY_CITY)
        last_weather_time = now_time
        if weather_data and weather_data.get("success"):
            weather_text = f"{weather_data['city']} {weather_data['temp']}C {weather_data['condition']}"
            print(f"Weather refreshed: {weather_text}")

# Draw hand landmarks

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS)
            
            # Landmark 8 get index finger tip
            lm8 = hand_landmarks.landmark[8]
            x8 = int(lm8.x * frame_width)
            y8 = int(lm8.y * frame_height)

            # Landmark 4 get thump tip
            lm4 = hand_landmarks.landmark[4]
            x4 = int(lm4.x * frame_width)
            y4 = int(lm4.y * frame_height)

            # draw dots on landmarks
            cv2.circle(frame, (x8, y8), 12, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x4, y4), 12, (255, 0, 0), cv2.FILLED)

            # show coordinates
            cv2.putText(frame, f"L8: ({x8},{y8})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"L4: ({x4},{y4})", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            #add current finger position to history
            now= time.time()
            finger_history.append((x8, y8, now))

            if len(finger_history) > SWIPE_HISTORY_SIZE:
                finger_history.pop(0)

            if now- last_swipe_time> SWIPE_COOLDOWN:
                swipe= detect_swipe(finger_history)

                if swipe is not None:
                    last_swipe_time= now
                    finger_history= [] #clear history so nwxt swipe starts fresh

                    in_top_zone = y8 < frame_height //2 #deciding the finger in top or bottom.

                    if in_top_zone:
                        top_index = (top_index + 1 if swipe == "right" else top_index - 1) % len(top_images)
                        swipe_feedback_text = f"TOP --> {top_files[top_index]}"
                    else:
                        bot_index = (bot_index + 1 if swipe == "right" else bot_index - 1) % len(bot_images)
                        swipe_feedback_text = f"BOTTOM --> {bot_files[bot_index]}"
 
                    swipe_feedback_timer = now

                    new_top    = top_files[top_index]
                    new_bot    = bot_files[bot_index]
                    new_vector = get_combination_vector(
                        top_features_dict[new_top],
                        bot_features_dict[new_bot]
                    )

                    api_save_combination(new_top, new_bot, new_vector)

                    status   = api_status()
                    db_total = status["total"]
                    db_liked = status["liked"]

            is_thumbs_up= detect_thumbs_up(hand_landmarks, frame_height)  #thumbs up logic

            if is_thumbs_up and (now- last_thumbs_up_time> THUMBS_UP_COOLDOWN):
                last_thumbs_up_time= now

                current_top_name = top_files[top_index]
                current_bot_name = bot_files[bot_index]



                already_liked= any(
                    c["top"] == current_top_name and c["bottom"] == current_bot_name
                    for c in liked_combinations
                )


                if already_liked:
                    thumbs_up_feedback_text  = "Already liked!"
                    thumbs_up_feedback_timer = now
                else:
                    liked_combinations.append({
                        "top": current_top_name,
                        "bottom": current_bot_name
                    })
                    thumbs_up_feedback_text  = f"LIKED! {current_top_name} + {current_bot_name}"
                    thumbs_up_feedback_timer = now
 
                    result = api_like(current_top_name, current_bot_name)
                    if result:
                        print(f"LIKED: {current_top_name} + {current_bot_name}")
                        print(f"New model accuracy: {result.get('new_accuracy', 0):.1%}")
 
                    status   = api_status()
                    db_total = status["total"]
                    db_liked = status["liked"]

    else:
        finger_history = []  # Clear history if no hand is detected

             
    # Overlay top-bottom outfit
    frame = overlay_png(frame, top_images[top_index], TOP_X, TOP_Y)
    frame = overlay_png(frame, bot_images[bot_index], BOT_X, BOT_Y)

    # Show current outfit names on screen
    cv2.putText(frame, f"Top: {top_files[top_index]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Bottom: {bot_files[bot_index]}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
   
    cv2.putText(frame, weather_text,
                (frame_width - 320, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (32, 0, 128), 2)

    match_percent = int(match_score * 100)
 
    match_color = (139, 0, 0)
 
    match_text = f"Match: {match_percent}%"
 
    cv2.putText(frame, match_text,
                (10, frame_height - 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, match_color, 2)


    # show swipe feedback mssg

    if swipe_feedback_text and time.time() - swipe_feedback_timer < 1.0:
        cv2.putText(frame, f"SWIPE: {swipe_feedback_text}",
                    (10, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        
    if thumbs_up_feedback_text and time.time() - thumbs_up_feedback_timer < 1.5:
        cv2.putText(frame, thumbs_up_feedback_text,
                    (10, frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
    
    cv2.putText(frame, f"DB: {db_total} seen | {db_liked} liked",
            (frame_width - 220, frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


    # Show liked counter in top right corner
    cv2.putText(frame, f"Liked: {len(liked_combinations)}",
                (frame_width - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (139, 0, 0), 2)        

    cv2.line(frame, (0, frame_height // 2), (frame_width, frame_height // 2),
             (100, 100, 100), 1)
    
    cv2.imshow("StyleMate - Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Camera closed.")
