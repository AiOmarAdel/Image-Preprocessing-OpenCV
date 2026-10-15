from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class ImageWithValueRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    value: float = Field(100, description="Value to apply")


@router.post("/add")
def add_value(req: ImageWithValueRequest):
    """Add a constant value to all pixels."""
    img = decode_image(req.image)
    result = np.clip(img.astype(np.float32) + req.value, 0, 255).astype(np.uint8)
    return image_response(result)


@router.post("/subtract")
def subtract_value(req: ImageWithValueRequest):
    """Subtract a constant value from all pixels."""
    img = decode_image(req.image)
    result = np.clip(img.astype(np.float32) - req.value, 0, 255).astype(np.uint8)
    return image_response(result)


@router.post("/divide")
def divide_value(req: ImageWithValueRequest):
    """Divide all pixels by a constant value."""
    if req.value == 0:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Division by zero is not allowed")
    img = decode_image(req.image)
    result = np.clip(img.astype(np.float32) / req.value, 0, 255).astype(np.uint8)
    return image_response(result)


@router.post("/complement")
def complement(req: ImageRequest):
    """Compute the complement (negative) of the image."""
    img = decode_image(req.image)
    result = 255 - img
    return image_response(result)
