from flask import Flask, render_template, url_for, Response, request, jsonify
import cv2
from deepface import DeepFace
from mtcnn import MTCNN
import numpy as np
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import threading
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__, static_url_path='/static')

# Emotion categories that DeepFace typically outputs
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

cam = cv2.VideoCapture(0)
detector = MTCNN()

max_emotion = None
max_count = 0
lock = threading.Lock()

os.environ['GOOGLE_API_KEY'] = api_key

def predict_emotion(face_image):
    try:
        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
        return analysis[0]['dominant_emotion']
    except Exception as e:
        print("Emotion analysis error:", e)
        return "neutral"

def detection():
    face_images, capture_interval = [], 1
    start_time = time.time()
    global max_emotion, max_count

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)

        if faces:
            face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = face['box']
            x, y = abs(x), abs(y)

            def expand_roi(x, y, w, h, scale_w, scale_h, img_shape):
                new_x = max(int(x - w * (scale_w - 1) / 2), 0)
                new_y = max(int(y - h * (scale_h - 1) / 2), 0)
                new_w = min(int(w * scale_w), img_shape[1] - new_x)
                new_h = min(int(h * scale_h), img_shape[0] - new_y)
                return new_x, new_y, new_w, new_h

            new_x, new_y, new_w, new_h = expand_roi(x, y, w, h, 1.3, 1.5, frame.shape)
            roi_color = frame[new_y:new_y + new_h, new_x:new_x + new_w]

            if time.time() - start_time >= capture_interval:
                face_images.append(roi_color)
                face_images = face_images[-5:]  # Keep last 5 frames
                start_time = time.time()

            emotion_counts = dict.fromkeys(emotion_labels, 0)
            for face in face_images:
                emotion = predict_emotion(face).lower()
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1

            with lock:
                max_emotion = max(emotion_counts, key=emotion_counts.get)
                max_count = emotion_counts[max_emotion]

            display_emotion = "neutral" if max_emotion == "surprise" else max_emotion
            cv2.putText(frame, display_emotion.capitalize(), (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if cv2.waitKey(1) & 0xFF == 13:
            break

        ret, buffer = cv2.imencode('.png', frame)
        yield (b'--frame\r\nContent-Type: image/png\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cam.release()
    cv2.destroyAllWindows()

def build_prompt_template(mood):
    common = "You are SoWell, a friendly and emotionally intelligent virtual therapist. A person is feeling {current_mood}. They asked: {topic}. Respond in a way that is calming, understanding, and helpful."

    mood_style = {
        "happy": "Celebrate their joy and offer ideas to enhance positivity.",
        "sad": "Empathize with their sadness, uplift them gently.",
        "angry": "Help them process anger constructively and provide soothing advice.",
        "fear": "Offer reassurance, focus on safety and clarity.",
        "disgust": "Help them reflect, and understand their feelings in a respectful tone.",
        "neutral": "Respond informatively and constructively.",
        "surprise": "Maintain a calm and grounding response."
    }

    addition = mood_style.get(mood.lower(), "")
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
        "happy": "😄 Hey, you look Happy today! Keep smiling!",
        "sad": "😢 Feeling a bit down? I’m here to cheer you up.",
        "angry": "😠 Uh-oh! Let’s take a deep breath together.",
        "fear": "😨 Everything’s going to be okay. Let’s talk it out.",
        "disgust": "😕 Hmm… Something bothering you?",
        "surprise": "😲 Whoa! You look surprised!",
        "neutral": "😐 You seem calm. That’s totally cool!"
    }
    return greetings.get(emotion.lower(), "👋 Hello there!")

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
 