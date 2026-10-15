import cv2
import numpy as np
import base64
from fastapi import HTTPException


def decode_image(base64_str: str) -> np.ndarray:
    """Decode a base64 image string to a numpy array (BGR)."""
    try:
        # Remove data URL prefix if present (e.g. "data:image/png;base64,...")
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


def encode_image(image: np.ndarray, fmt: str = ".png") -> str:
    """Encode a numpy array image to a base64 string."""
    success, buffer = cv2.imencode(fmt, image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def image_response(image: np.ndarray, fmt: str = ".png") -> dict:
    """Return a standard image response with base64 encoded image."""
    return {
        "image": encode_image(image, fmt),
        "format": fmt.lstrip("."),
        "width": image.shape[1],
        "height": image.shape[0],
    }
