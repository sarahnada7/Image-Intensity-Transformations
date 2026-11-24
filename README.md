# Image Intensity Transformations

## 📌 Project Overview
This project demonstrates different **image intensity transformations** using Python. These techniques are used in Digital Image Processing to modify pixel intensity values, enhance image details, and improve visual appearance.

## 🔍 Implemented Transformations

The project covers three main categories:

### 1. Linear Intensity Transformations
- **Identity Transformation:** Keeps the original pixel values unchanged.
- **Negative Transformation:** Inverts pixel intensities to create a photographic negative.

### 2. Logarithmic Transformation
- Enhances details in darker regions of an image.
- Compresses higher intensity values to improve visibility.

**Formula:**  
\(s = c \log_{10}(1+r)\)

### 3. Power-Law (Gamma) Transformation
- Controls image brightness using different gamma values.
- Gamma < 1: Brightens the image.
- Gamma = 1: No change.
- Gamma > 1: Darkens the image.

**Formula:**  
\(s = c \times r^\gamma\)

## 🛠️ Technologies Used
- Python
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## 📂 Project Structure

```text
Image-Intensity-Transformations/
│
├── python linear_transformations.py.py
├── log_transformation.py
├── gamma_transformation.py
└── README.md
```

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. Install the required libraries:
   ```bash
   pip install opencv-python numpy matplotlib
   ```
3. Place the input images in the project directory.
4. Run any of the Python scripts:
   ```bash
   python log_transformation.py
   ```

## 🎯 Objective
The objective of this project is to understand and implement fundamental image intensity transformation techniques and observe how different mathematical functions affect image brightness and contrast.

---

**Course:** Digital Image Processing
