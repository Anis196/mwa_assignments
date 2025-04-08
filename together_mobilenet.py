from flask import Flask, render_template, url_for, Response, request, jsonify
import cv2
from keras.models import load_model
import numpy as np
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from tensorflow.keras.utils import img_to_array
import threading
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__, static_url_path='/static')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('mobile_net_v2_firstmodel.h5', compile=False)

max_emotion = None
max_count = 0
lock = threading.Lock()

os.environ['GOOGLE_API_KEY'] = api_key

def predict_emotion(face_image):
    face_image = cv2.imdecode(np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR)
    face_image = cv2.resize(face_image, (224, 224)) / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    predictions = model.predict(face_image)
    return emotion_labels[np.argmax(predictions)]

def detection():
    face_images, capture_interval = [], 1
    start_time = time.time()
    global max_emotion, max_count

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:

            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

            def expand_roi(x, y, w, h, scale_w, scale_h, img_shape):
                new_x = max(int(x - w * (scale_w - 1) / 2), 0)
                new_y = max(int(y - h * (scale_h - 1) / 2), 0)
                new_w = min(int(w * scale_w), img_shape[1] - new_x)
                new_h = min(int(h * scale_h), img_shape[0] - new_y)
                return new_x, new_y, new_w, new_h

            new_x, new_y, new_w, new_h = expand_roi(x, y, w, h, 1.3, 1.5, frame.shape)
            roi_color = frame[new_y:new_y + new_h, new_x:new_x + new_w]

            if time.time() - start_time >= capture_interval:
                face_images.append(cv2.imencode('.png', roi_color)[1].tobytes())
                face_images = face_images[-5:]  # Keep last 5
                start_time = time.time()

            emotion_counts = dict.fromkeys(emotion_labels, 0)
            for face in face_images:
                emotion_counts[predict_emotion(face)] += 1

            with lock:
                max_emotion = max(emotion_counts, key=emotion_counts.get)
                max_count = emotion_counts[max_emotion]

            display_emotion = "Neutral" if max_emotion == "Surprise" else max_emotion
            cv2.putText(frame, display_emotion, (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

        if cv2.waitKey(1) & 0xFF == 13:
            break

        ret, buffer = cv2.imencode('.png', frame)
        yield (b'--frame\r\nContent-Type: image/png\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()

def build_prompt_template(mood):
    common = "You are SoWell, a friendly and emotionally intelligent virtual therapist. A person is feeling {current_mood}. They asked: {topic}. Respond in a way that is calming, understanding, and helpful."
    
    # Optional: add unique tone per mood
    mood_style = {
        "Happy": "Celebrate their joy and offer ideas to enhance positivity.",
        "Sad": "Empathize with their sadness, uplift them gently.",
        "Angry": "Help them process anger constructively and provide soothing advice.",
        "Fear": "Offer reassurance, focus on safety and clarity.",
        "Disgust": "Help them reflect, and understand their feelings in a respectful tone.",
        "Neutral": "Respond informatively and constructively.",
        "Surprise": "Maintain a calm and grounding response."
    }

    addition = mood_style.get(mood, "")
    return PromptTemplate(
        input_variables=["topic", "current_mood"],
        template=f"{common} {addition}"
    )

def initialize_bot(current_mood):
    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
    prompt = build_prompt_template(current_mood)
    return LLMChain(
        llm=ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-latest",
            google_api_key=os.environ.get('GOOGLE_API_KEY'),
            temperature=0.7
        ),
        prompt=prompt
    )

def clean_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.replace('*', '')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text).replace('\t', ' ').strip()
    return text

def bot_answer(question, current_mood):
    chain = initialize_bot(current_mood)
    return clean_text(chain.run({"topic": question, "current_mood": current_mood}))

@app.route('/')
def about():
    return render_template('together.html')

def emotion_greeting(emotion):
    greetings = {
        "Happy": "😄 Hey, you look Happy today! Keep smiling!",
        "Sad": "😢 Feeling a bit down? I’m here to cheer you up.",
        "Angry": "😠 Uh-oh! Let’s take a deep breath together.",
        "Fear": "😨 Everything’s going to be okay. Let’s talk it out.",
        "Disgust": "😕 Hmm… Something bothering you?",
        "Surprise": "😲 Whoa! You look surprised!",
        "Neutral": "😐 You seem calm. That’s totally cool!"
    }
    return greetings.get(emotion, "👋 Hello there!")

@app.route('/submit', methods=['POST'])
def submit():
    with lock:
        emotion = max_emotion
    response = bot_answer("Who are you?", emotion)
    greeting = emotion_greeting(emotion)
    return render_template('together.html', emotion=emotion, response=response, greeting=greeting)

@app.route('/video', methods=['GET'])
def video():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    with lock:
        current_emotion = max_emotion
    bot_response = bot_answer(user_message, current_emotion)
    return jsonify({'bot_message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
