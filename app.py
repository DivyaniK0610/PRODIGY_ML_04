import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data', '00')
IMG_SIZE = 64

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- EMOJI MAPPING ---
# We map each folder name to a (Name, Emoji) pair
gesture_map = {
    '01_palm':       ('Palm', '✋'),
    '02_l':          ('L Sign', '👆'),
    '03_fist':       ('Fist', '✊'),
    '04_fist_moved': ('Fist', '✊'),
    '05_thumb':      ('Thumb Up', '👍'),
    '06_index':      ('Index Finger', '☝️'),
    '07_ok':         ('OK Sign', '👌'),
    '08_palm_moved': ('Palm', '✋'),
    '09_c':          ('C Sign', '🤏'),
    '10_down':       ('Down', '👇')
}

# --- 1. TRAIN MODEL ON STARTUP ---
print("😃 Starting Emoji Mirror... Training Model...")
data = []
labels = []

if os.path.exists(DATA_DIR):
    categories = os.listdir(DATA_DIR)
    for category in categories:
        path = os.path.join(DATA_DIR, category)
        if os.path.isdir(path):
            # Load 50 images per category for fast startup
            for img_name in os.listdir(path)[:50]:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img.flatten())
                    labels.append(category)

X = np.array(data)
y = np.array(labels)

model = SVC(kernel='poly', degree=3, probability=True)
if len(X) > 0:
    model.fit(X, y)
    print("✅ Model Trained! Ready to match emojis.")
else:
    print("❌ Error: No data found in data/00")

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_emoji', methods=['POST'])
def predict_emoji():
    if 'file' not in request.files:
        return jsonify(error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected")

    if file:
        # Save file temporarily
        filename = 'temp_upload.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process Image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        vec = img.flatten().reshape(1, -1)

        # Predict
        prediction = model.predict(vec)[0]
        
        # Get Name and Emoji
        result = gesture_map.get(prediction, ("Unknown", "❓"))
        gesture_name = result[0]
        gesture_emoji = result[1]

        return jsonify(
            name=gesture_name, 
            emoji=gesture_emoji, 
            image_url=url_for('static', filename=f'uploads/{filename}')
        )

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Using port 8080 to avoid conflicts
    app.run(host='0.0.0.0', port=8080, debug=False)