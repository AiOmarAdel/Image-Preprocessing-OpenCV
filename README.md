# Image Preprocessing & Computer Vision with OpenCV

A hands-on project exploring fundamental image preprocessing and computer vision techniques using Python and OpenCV. This project demonstrates how images can be processed, enhanced, and analyzed as a foundation for more advanced applications such as object detection and image analysis.

## 🚀 Features

- **Image Processing:** Image loading, displaying, resizing, and manipulation.
- **Color Space Conversion:** Working with BGR, RGB, Grayscale, and HSV color spaces.
- **Image Blurring:** Applying Gaussian Blur and other smoothing techniques.
- **Thresholding:** Binary and adaptive thresholding for image segmentation.
- **Edge Detection:** Detecting image edges using the Canny algorithm.
- **Contour Detection:** Identifying and drawing object contours.
- **Color Detection:** Detecting specific colors using HSV ranges and masking.
- **Real-Time Processing:** Applying color detection to webcam video streams.

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Jupyter Notebook

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Navigate to the project directory:

   ```bash
   cd Image-Preprocessing-OpenCV
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the Python scripts or open the Jupyter Notebook to explore the different image preprocessing techniques.

```python
import cv2

image = cv2.imread("images/sample.jpg")

if image is not None:
    resized_image = cv2.resize(image, (640, 480))
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    cv2.imshow("Original Image", image)
    cv2.imshow("Processed Image", blurred_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error: Could not load the image.")
```

**Note:** Replace `YOUR_REPOSITORY_URL` with your actual GitHub repository URL. Update the image path to match your project files.

## 🎯 Project Goals

- Understand fundamental image processing operations.
- Explore how preprocessing techniques affect image quality and visual features.
- Build a foundation for Computer Vision and Object Detection projects.
- Develop practical skills using Python and OpenCV.

## 🔮 Future Improvements

- Add image enhancement and restoration techniques.
- Explore morphological operations and image segmentation.
- Build an interactive image processing web application.
- Integrate preprocessing pipelines into object detection models.

## 👨‍💻 Author

**Omar Adel Abdul Hakam**

AI Student | Computer Vision | Machine Learning | Deep Learning

## 📄 License

This project is intended for educational and learning purposes.
