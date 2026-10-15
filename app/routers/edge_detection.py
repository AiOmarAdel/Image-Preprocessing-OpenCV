from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, encode_image
import cv2
import numpy as np

router = APIRouter()

Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = kernel.shape[0] // 2
    padded = np.pad(image, ((k, k), (k, k)), mode='constant')
    output = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i + 2 * k + 1, j:j + 2 * k + 1]
            output[i, j] = np.sum(region * kernel)
    return output


@router.post("/sobel")
def sobel_edge_detection(req: ImageRequest):
    """
    Apply Sobel edge detection.
    Returns: combined, grad_x, grad_y, grad_avg — all as base64 images.
    """
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    Gx = _convolve(gray, Kx)
    Gy = _convolve(gray, Ky)

    sobel_combined = np.clip(np.sqrt(Gx ** 2 + Gy ** 2), 0, 255).astype(np.uint8)
    abs_gx = np.abs(Gx).astype(np.uint8)
    abs_gy = np.abs(Gy).astype(np.uint8)
    grad_avg = (abs_gx * 0.5 + abs_gy * 0.5).astype(np.uint8)

    def to_bgr_b64(arr):
        return encode_image(cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))

    return {
        "combined": to_bgr_b64(sobel_combined),
        "grad_x": to_bgr_b64(abs_gx),
        "grad_y": to_bgr_b64(abs_gy),
        "grad_avg": to_bgr_b64(grad_avg),
        "width": img.shape[1],
        "height": img.shape[0],
    }
