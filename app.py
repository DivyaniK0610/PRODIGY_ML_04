import os
import cv2
import numpy as np
from flask import Flask, render_template, request, url_for
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DATA_DIR = os.path.join(BASE_DIR, 'data', '00')

IMG_SIZE = 64

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

gesture_names = {
    '01_palm': 'Palm ✋',
    '02_l': 'L Sign 👆',
    '03_fist': 'Fist ✊',
    '04_fist_moved': 'Fist (Moved) ✊',
    '05_thumb': 'Thumb Up 👍',
    '06_index': 'Index Finger ☝️',
    '07_ok': 'OK Sign 👌',
    '08_palm_moved': 'Palm (Moved) ✋',
    '09_c': 'C Sign 🤏',
    '10_down': 'Down 👇'
}

data = []
labels = []

if os.path.exists(DATA_DIR):
    categories = os.listdir(DATA_DIR)
    for category in categories:
        folder_path = os.path.join(DATA_DIR, category)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img.flatten())
                    labels.append(category)

X = np.array(data)
y = np.array(labels)

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='poly', degree=3) 
    model.fit(X_train, y_train)
    print(f"Model Trained. Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")
else:
    print("Error: No images found. Check data/00 folder structure.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_url = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                vec = img.flatten().reshape(1, -1)
                pred_label = model.predict(vec)[0]
                prediction = gesture_names.get(pred_label, pred_label)
                img_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', prediction=prediction, img_url=img_url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, port=5004)