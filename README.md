# Image Denoising

A comprehensive implementation of various **Image Denoising** techniques to restore high-quality images from noise-contaminated versions. This repository explores both classical computer vision filters and modern deep learning-based approaches to suppress noise while preserving critical image edges and details.

## 📖 Overview

Image noise can be caused by sensor limitations or poor environmental conditions. The goal of this project is to estimate the original image  from a noisy observation , where  represents the additive noise.

### Key Highlights:

* **Edge Preservation:** Techniques designed to reduce noise without blurring significant details.
* **Multiple Noise Types:** Support for Gaussian, Salt & Pepper, and Poisson noise removal.
* **Extensible Framework:** Easily add and test new denoising algorithms.

## 🚀 Features

* [x] **Classical Filters:** Median, Gaussian, and Bilateral filtering.
* [x] **Non-Local Means (NLM):** Advanced patch-based averaging for superior detail preservation.
* [x] **Deep Learning (Optional):** Pre-trained models for state-of-the-art restoration.
* [x] **Visualization:** Side-by-side comparison of noisy vs. denoised results.

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ChetanKothawale/Image_Denoising.git
cd Image_Denoising

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## 💻 Usage

To denoise a single image using the default algorithm:

```bash
python denoise.py --input path/to/noisy_image.jpg --method nlm --output result.png

```

### Supported Methods:

* `gaussian`: Traditional Gaussian blurring.
* `median`: Effective for Salt & Pepper noise.
* `nlm`: Non-Local Means (Recommended for natural images).
* `bilateral`: Edge-preserving smoothing.

## 📊 Results

| Noisy Image | Denoised (NLM) | Denoised (Deep Learning) |
| --- | --- | --- |
|  |  |  |

## 📚 References

* [Non-Local Means Algorithm](https://en.wikipedia.org/wiki/Non-local_means)
* [OpenCV Photo Denoising Documentation](https://www.google.com/search?q=https://docs.opencv.org/master/d1/dfd/group__photo__render.html)


---

[Denoising Images with OpenCV](https://www.youtube.com/watch?v=xtRY_iT41U4)

This video provides a practical guide on implementing various image denoising algorithms in Python using OpenCV, which aligns perfectly with the core goals of your repository.
