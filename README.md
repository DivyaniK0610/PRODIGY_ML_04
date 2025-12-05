# Hand Gesture Recognition Engine (PRODIGY_ML_04)

**A real-time computer vision system that interprets human hand gestures using a lightweight Support Vector Machine (SVM) architecture.**

![Project Screenshot](images/demo_screenshot.png)

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


