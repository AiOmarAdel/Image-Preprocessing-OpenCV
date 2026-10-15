from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import cv2
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class KernelRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    kernel_size: int = Field(3, ge=1, le=21, description="Kernel size (odd number)")


def _preprocess_binary(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
    return binary


@router.post("/dilation")
def apply_dilation(req: KernelRequest):
    """Apply morphological dilation to a binary image."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.dilate(binary, kernel, iterations=1)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))


@router.post("/erosion")
def apply_erosion(req: KernelRequest):
    """Apply morphological erosion to a binary image."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.erode(binary, kernel, iterations=1)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))


@router.post("/opening")
def apply_opening(req: KernelRequest):
    """Apply morphological opening (erosion then dilation)."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))


@router.post("/boundary/internal")
def internal_boundary(req: KernelRequest):
    """Extract the internal boundary of objects in a binary image."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.subtract(binary, cv2.erode(binary, kernel))
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))


@router.post("/boundary/external")
def external_boundary(req: KernelRequest):
    """Extract the external boundary of objects in a binary image."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.subtract(cv2.dilate(binary, kernel), binary)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))


@router.post("/gradient")
def morphological_gradient(req: KernelRequest):
    """Compute the morphological gradient (dilation - erosion)."""
    img = decode_image(req.image)
    binary = _preprocess_binary(img)
    k = req.kernel_size
    kernel = np.ones((k, k), np.uint8)
    result = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
