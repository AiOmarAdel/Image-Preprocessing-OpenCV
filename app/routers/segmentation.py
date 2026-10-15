from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import cv2
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class ThresholdRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    thresh_value: int = Field(130, ge=0, le=255, description="Threshold value (0-255)")


@router.post("/basic-global")
def basic_global_thresholding(req: ThresholdRequest):
    """Apply basic global thresholding."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, req.thresh_value, 255, cv2.THRESH_BINARY)
    return image_response(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))


@router.post("/otsu")
def automatic_thresholding(req: ThresholdRequest):
    """Apply Otsu's automatic thresholding."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, req.thresh_value, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return image_response(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))


@router.post("/adaptive")
def adaptive_thresholding(req: ImageRequest):
    """Apply adaptive mean thresholding."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 199, 5
    )
    return image_response(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))


@router.post("/vertical-otsu")
def vertical_adaptive_otsu(req: ImageRequest):
    """Apply Otsu's thresholding to vertical segments and concatenate results."""
    img = decode_image(req.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    parts = [gray[i * height // 4:(i + 1) * height // 4, :] for i in range(4)]
    segmented = [cv2.threshold(p, 0, 255, cv2.THRESH_OTSU)[1] for p in parts]
    result = np.concatenate(segmented, axis=0)
    return image_response(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))
