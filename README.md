# Image Preprocessing & Computer Vision Studio

A Python-based Computer Vision project built with OpenCV, designed to explore and apply essential image processing techniques through a modular and organized application.

The project provides tools for image filtering, color processing, edge detection, histogram analysis, morphological operations, image restoration, and segmentation.

## ✨ Features

### 🎨 Color Processing
- Color space conversion and image color manipulation.
- RGB, BGR, Grayscale, and HSV processing.
- HSV-based color detection and masking.

### 🖼️ Image Filtering
- Gaussian Blur and image smoothing.
- Noise reduction using different filtering techniques.
- Image sharpening and detail enhancement.

### 📈 Histogram Analysis
- Image histogram calculation and visualization.
- Contrast enhancement techniques.
- Histogram Equalization and related operations.

### 🔍 Edge Detection
- Canny Edge Detection.
- Edge extraction for image analysis.
- Detection of important visual boundaries.

### 🧩 Morphological Operations
- Erosion and Dilation.
- Opening and Closing.
- Morphological image processing for noise removal and shape analysis.

### 🎯 Point Operations
- Pixel intensity transformations.
- Brightness and contrast adjustment.
- Basic image enhancement operations.

### 🛠️ Image Restoration
- Image denoising and noise reduction.
- Restoration and enhancement techniques.
- Improving image quality for further processing.

### 🗂️ Image Segmentation
- Threshold-based segmentation.
- Separating foreground and background regions.
- Preparing images for object and feature analysis.

## 🏗️ Project Structure

```text
Image-Preprocessing-OpenCV/
├── app/
│   ├── main.py
│   ├── utils.py
│   └── routers/
│       ├── color.py
│       ├── edge_detection.py
│       ├── filters.py
│       ├── histogram.py
│       ├── morphology.py
│       ├── point_ops.py
│       ├── restoration.py
│       └── segmentation.py
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- **Python** — Core programming language.
- **OpenCV** — Image processing and computer vision operations.
- **NumPy** — Numerical computations and image array manipulation.
- **FastAPI** — Application backend and API routing, if used by the project.
- **Git & GitHub** — Version control and project hosting.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Image-Preprocessing-OpenCV
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

If the project uses FastAPI with an application instance named `app` in `app/main.py`, run:

```bash
uvicorn app.main:app --reload
```

Open the API documentation in your browser:

```text
http://127.0.0.1:8000/docs
```

Adjust the command if your application uses a different framework or entry point.

## 🎯 Project Objectives

- Build a practical understanding of image processing fundamentals.
- Explore classical Computer Vision algorithms.
- Understand how preprocessing improves image analysis.
- Organize image processing techniques into reusable modules.
- Establish a foundation for advanced Computer Vision and Object Detection projects.

## 🚀 Future Improvements

- Add contour detection and shape recognition.
- Implement geometric transformations and perspective correction.
- Add real-time webcam processing.
- Integrate YOLO for object detection.
- Include visual comparisons between original and processed images.
- Add automated tests and performance evaluation.

## 👨‍💻 Author

**Omar Adel Abdul Hakam**

AI Student | Computer Vision | Machine Learning | Deep Learning

## 📄 License

Specify the license under which this project is distributed. If you intend to publish it as an open-source project, consider adding an appropriate license file.
