from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import cv2
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class SaltPepperRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    prob: float = Field(0.04, ge=0.0, le=1.0, description="Noise probability (0.0 - 1.0)")


class GaussianNoiseRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    mean: float = Field(0, description="Gaussian mean")
    std: float = Field(20, description="Gaussian standard deviation")


class OutlierRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded grayscale image")
    threshold: int = Field(30, description="Outlier threshold")


class AveragingRequest(BaseModel):
    image1: str = Field(..., description="Base64 encoded first image")
    image2: str = Field(..., description="Base64 encoded second image")


# ── Noise ──────────────────────────────────────────────────────────────────────

@router.post("/noise/salt-pepper")
def add_salt_pepper(req: SaltPepperRequest):
    """Add salt-and-pepper noise to the image."""
    img = decode_image(req.image)
    noisy = img.copy()
    probs = np.random.rand(*img.shape[:2])
    noisy[probs < req.prob / 2] = 0
    noisy[probs > 1 - req.prob / 2] = 255
    return image_response(noisy)


@router.post("/noise/gaussian")
def add_gaussian_noise(req: GaussianNoiseRequest):
    """Add Gaussian noise to the image."""
    img = decode_image(req.image)
    gauss = np.random.normal(req.mean, req.std, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), gauss)
    return image_response(np.clip(noisy, 0, 255).astype(np.uint8))


# ── Filters ────────────────────────────────────────────────────────────────────

@router.post("/filter/average")
def average_filter(req: ImageRequest):
    """Apply 3x3 average filter to a grayscale image."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            result[i, j] = np.mean(padded[i:i+3, j:j+3])
    return image_response(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR))


@router.post("/filter/median")
def median_filter(req: ImageRequest):
    """Apply 3x3 median filter to a grayscale image."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            result[i, j] = np.median(padded[i:i+3, j:j+3])
    return image_response(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR))


@router.post("/filter/outlier")
def outlier_filter(req: OutlierRequest):
    """Apply outlier filter — replaces outlier pixels with local mean."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result = gray.copy()
    for i in range(1, gray.shape[0] + 1):
        for j in range(1, gray.shape[1] + 1):
            window = padded[i-1:i+2, j-1:j+2]
            center = float(padded[i, j])
            mean = float(np.mean(window))
            if abs(center - mean) > req.threshold:
                result[i-1, j-1] = int(mean)
    return image_response(cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR))


@router.post("/filter/averaging")
def image_averaging(req: AveragingRequest):
    """Average two images to reduce noise."""
    img1 = decode_image(req.image1)
    img2 = decode_image(req.image2)
    avg = (img1.astype(np.float32) + img2.astype(np.float32)) / 2
    return image_response(avg.astype(np.uint8))
