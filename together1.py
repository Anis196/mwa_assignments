from flask import Flask, render_template, url_for, Response, request, jsonify
import cv2
from keras.models import load_model
import numpy as np
import time
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
import re
from tensorflow.keras.utils import img_to_array
import threading
import os
from langchain.memory import ConversationBufferMemory


app = Flask(__name__,static_url_path='/static')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Opens Camera
cam = cv2.VideoCapture(0)

# Loading the face detection and the emotion classification models
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('mobile_net_v2_firstmodel.h5', compile=False)
max_emotion = None
max_count = 0
lock = threading.Lock()


def predict_emotion(face_image):
    face_image = cv2.imdecode(np.frombuffer(face_image, np.uint8), cv2.IMREAD_COLOR)
    final_image = cv2.resize(face_image, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0

    predictions = model.predict(final_image)

    predicted_emotion = emotion_labels[np.argmax(predictions)]

    return predicted_emotion

def detection():
    face_images = []
    capture_interval = 1
    start_time = time.time()
    


    global max_count, max_emotion

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = largest_face

            def expand_roi(x, y, w, h, scale_w, scale_h, img_shape):
                new_x = max(int(x - w * (scale_w - 1) / 2), 0)
                new_y = max(int(y - h * (scale_h - 1) / 2), 0)
                new_w = min(int(w * scale_w), img_shape[1] - new_x)
                new_h = min(int(h * scale_h), img_shape[0] - new_y)
                return new_x, new_y, new_w, new_h

            scale_w = 1.3
            scale_h = 1.5

            new_x, new_y, new_w, new_h = expand_roi(x, y, w, h, scale_w, scale_h, frame.shape)
            roi_color = frame[new_y:new_y+new_h, new_x:new_x+new_w]

            if time.time() - start_time >= capture_interval:
                face_images.append(cv2.imencode('.png', roi_color)[1].tobytes())
                if len(face_images) > 5:
                    face_images.pop(0)
                start_time = time.time()
                
            emotion_counts = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Surprise": 0, "Sad": 0, "Neutral": 0}
            if len(face_images) >= 4:
                for face_image in face_images:
                    predicted_emotion = predict_emotion(face_image)
                    emotion_counts[predicted_emotion] += 1

            with lock:
                max_emotion = max(emotion_counts, key=emotion_counts.get)
                max_count = emotion_counts[max_emotion]

            status = max_emotion
            if status=="Surprise":
                status="Neutral"
            cv2.putText(frame, status, (100, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))   
                    #face_images=[]

        if cv2.waitKey(1) & 0xFF == 13:
            break
        
        ret, buffer = cv2.imencode('.png', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
    cam.release()
    cv2.destroyAllWindows()

os.environ['GOOGLE_API_KEY'] = "AIzaSyBCtxI3nT607F8GWE_pkwbSSYXHRQJNfA4"

def initialize_bot(current_mood):
    from langchain_google_genai import ChatGoogleGenerativeAI  # make sure this import is present

    api_key = os.environ.get('GOOGLE_API_KEY')  # safer and more common practice

    genai.configure(api_key=api_key)

    if current_mood == 'Happy' or current_mood == 'Surprise':
        title_template = PromptTemplate(
            input_variables=['topic', 'current_mood'],
            template='You are a Personal Therapist named SoWell. ... joyful mood ... query: {topic} ...')
    elif current_mood == 'Angry' or current_mood == 'Disgust':
        title_template = PromptTemplate(
            input_variables=['topic', 'current_mood'],
            template='You are a Personal Therapist named SoWell. ... angry mood ... query: {topic} ...')
    elif current_mood == 'Fear' or current_mood == 'Sad':
        title_template = PromptTemplate(
            input_variables=['topic', 'current_mood'],
            template='You are a Personal Therapist named SoWell. ... sad mood ... query: {topic} ...')
    else:
        title_template = PromptTemplate(
            input_variables=['topic', 'current_mood'],
            template='You are a Personal Therapist named SoWell. ... general mood ... query: {topic} ...')

    # ✅ Use API-key auth method
    llm_global = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro-latest",
        google_api_key=api_key,
        temperature=0.7
    )

    return LLMChain(
        llm=llm_global,
        prompt=title_template
    )

    
def clean_text(text):
    # Handle None or non-string input
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove asterisks
    cleaned_text = text.replace('*', '')
    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    # Replace tab characters with spaces
    cleaned_text = cleaned_text.replace('\t', ' ')
    # Trim leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def bot_answer(question, current_mood):
    llm_chain = initialize_bot(current_mood)
    result = llm_chain.run({"topic": question, "current_mood": current_mood})
    return clean_text(result)


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
    global max_emotion
    with lock:
        emotion = max_emotion
    response = bot_answer('Who are you?', emotion)
    greeting = emotion_greeting(emotion)
    return render_template('together.html', emotion=emotion, response=response, greeting=greeting)


@app.route('/video', methods=['GET', 'POST'])
def video():
    return Response(detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    with lock:
        current_emotion = max_emotion
    bot_response = bot_answer(user_message, current_emotion)
    return jsonify({'bot_message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
