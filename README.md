# Hand Gesture Recognition Engine (PRODIGY_ML_04)

**A real-time computer vision system that interprets human hand gestures using a lightweight Support Vector Machine (SVM) architecture.**

![Project Demo](images/hand.png)
---

## 🚀 Innovation & Approach
Hand gesture recognition is typically solved using heavy Convolutional Neural Networks (CNNs) that require GPUs. This project takes a **resource-efficient approach** by implementing a **Polynomial SVM Classifier**. By focusing on structural feature extraction rather than deep convolution, this engine delivers high-accuracy gesture recognition on standard CPUs with millisecond latency.

### Key Features
1.  **Zero-Lag Web Interface:** A highly responsive **Flask** application allows users to upload gesture images and receive instant feedback, simulating a real-time control system.
2.  **Geometric Feature Analysis:** The system uses **OpenCV** to preprocess hand shapes (grayscale conversion, resizing) to isolate geometric features before classification.
3.  **High-Precision Classification:** Trained on the **LeapGestRecog** dataset, the model distinguishes between nuanced gestures (like "Fist" vs. "Fist Moved") with high confidence.

---

## 🛠️ Tech Stack
* **Python 3.x**
* **Scikit-Learn:** Support Vector Machine (SVM - Poly Kernel)
* **OpenCV (cv2):** Image Preprocessing & Computer Vision
* **Flask:** Web Server & API
* **NumPy:** High-performance Matrix Operations
* **HTML/CSS:** Modern User Interface

---

## 📂 Directory Structure

```text
PRODIGY_ML_04/
│
├── data/
│   └── 00/                 # Training Data (Subject 00 subset)
│       ├── 01_palm/
│       ├── 02_l/
│       └── ...
│
├── static/
│   ├── uploads/            # Temporary image storage
│   └── style.css           # UI Styling
│
├── templates/
│   └── index.html          # Frontend Interface
│
├── app.py                  # Main Recognition Engine
├── requirements.txt        # Dependencies
└── README.md               # Project Documentation


```
---

## ⚡ How to Run
1. Clone the repository (or download the folder).

2. nstall Dependencies:
```Bash
pip install -r requirements.txt
```
3. Setup Data:
Download the **LeapGestRecog Dataset**.
Extract the 00 folder (Subject 00) into the data/ directory so the path resolves to data/00/01_palm/....

4. Run the Engine:
```Bash
python app.py
```

5. View the Results:
The script will launch a server at http://127.0.0.1:8080.
Upload an image of your hand to see the Emoji Prediction.

---
## 📊 Recognizable Gestures

The engine is trained to identify 10 distinct hand commands, mapped to specific emojis:

| Gesture Name | Emoji Output | Description |
| :--- | :---: | :--- |
| **Palm** | ✋ | Open hand facing forward |
| **Fist** | ✊ | Closed fingers (Rock) |
| **Thumb Up** | 👍 | Thumb extended upward |
| **OK Sign** | 👌 | Thumb and index forming a circle |
| **Index** | ☝️ | Index finger pointing up |
| **L Sign** | 👆 | Thumb and index forming 'L' |
| **C Sign** | 🤏 | Hand forming a 'C' shape |
| **Down** | 👇 | Index pointing down |

---
## 📜 Dataset

* **Source:** Kaggle LeapGestRecog
* **Subset Used:** Subject 00 (Approx. 2,000 images).

### Preprocessing Pipeline
* **Grayscale Conversion:** Applied to reduce color noise and computational load.
* **Resizing:** Images are normalized to **64x64 pixels**.
* **Flattening:** Converted into **1D feature vectors** (4096 features) to serve as input for the SVM classifier.