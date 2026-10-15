from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import cv2
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


def _average_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    result = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                result[i, j, c] = np.sum(padded[i:i+3, j:j+3, c]) // 9
    return result


def _laplacian_filter(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    padded = np.pad(gray, 1, mode='edge')
    output = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            region = padded[i:i+3, j:j+3]
            val = np.sum(region * kernel)
            output[i, j] = np.clip(abs(val), 0, 255)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def _maximum_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                output[i, j, c] = np.max(padded[i:i+3, j:j+3, c])
    return output


def _minimum_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                output[i, j, c] = np.min(padded[i:i+3, j:j+3, c])
    return output


def _median_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                window = padded[i:i+3, j:j+3, c].flatten()
                output[i, j, c] = np.median(window)
    return output


def _mode_filter(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(3):
                window = padded[i:i+3, j:j+3, c].flatten()
                counts = np.bincount(window)
                output[i, j, c] = np.argmax(counts)
    return output


@router.post("/average")
def average_filter(req: ImageRequest):
    """Apply 3x3 average (mean) filter to smooth the image."""
    img = decode_image(req.image)
    return image_response(_average_filter(img))


@router.post("/laplacian")
def laplacian_filter(req: ImageRequest):
    """Apply 3x3 Laplacian filter for edge detection."""
    img = decode_image(req.image)
    return image_response(_laplacian_filter(img))


@router.post("/maximum")
def maximum_filter(req: ImageRequest):
    """Apply 3x3 maximum filter (dilation effect)."""
    img = decode_image(req.image)
    return image_response(_maximum_filter(img))


@router.post("/minimum")
def minimum_filter(req: ImageRequest):
    """Apply 3x3 minimum filter (erosion effect)."""
    img = decode_image(req.image)
    return image_response(_minimum_filter(img))


@router.post("/median")
def median_filter(req: ImageRequest):
    """Apply 3x3 median filter to reduce noise."""
    img = decode_image(req.image)
    return image_response(_median_filter(img))


@router.post("/mode")
def mode_filter(req: ImageRequest):
    """Apply 3x3 mode filter (most frequent value in window)."""
    img = decode_image(req.image)
    return image_response(_mode_filter(img))
