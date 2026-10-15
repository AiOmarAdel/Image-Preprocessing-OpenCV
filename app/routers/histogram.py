from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import cv2
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


def _hist_stretching(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val == min_val:
        return image
    stretched = (gray - min_val) * (255.0 / (max_val - min_val))
    return cv2.cvtColor(stretched.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _compute_histogram(ch_image: np.ndarray) -> list:
    hist = [0] * 256
    for i in range(ch_image.shape[0]):
        for j in range(ch_image.shape[1]):
            hist[ch_image[i, j]] += 1
    return hist


def _histogram_equalization(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = _compute_histogram(gray)
    cdf = [0] * 256
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    cdf_min = next((v for v in cdf if v > 0), 0)
    total_pixels = cdf[-1]
    if total_pixels == cdf_min:
        return image
    mapping = [
        round((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255.0)
        for i in range(256)
    ]
    r, c = gray.shape
    equalized = np.zeros((r, c), dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            equalized[i, j] = mapping[gray[i, j]]
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


@router.post("/stretch")
def hist_stretch(req: ImageRequest):
    """Perform histogram stretching to enhance contrast."""
    img = decode_image(req.image)
    result = _hist_stretching(img)
    return image_response(result)


@router.post("/equalize")
def hist_equalize(req: ImageRequest):
    """Perform histogram equalization to improve contrast."""
    img = decode_image(req.image)
    result = _histogram_equalization(img)
    return image_response(result)


@router.post("/values")
def get_histogram(req: ImageRequest):
    """Return the histogram values for each channel (B, G, R) and grayscale."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().tolist()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten().tolist()
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten().tolist()
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    return {
        "blue": hist_b,
        "green": hist_g,
        "red": hist_r,
        "gray": hist_gray,
    }
