from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.utils import decode_image, image_response
import numpy as np

router = APIRouter()


class ImageRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class ChannelRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    value: int = Field(100, description="Value to add/subtract from channel")


@router.post("/red")
def change_red(req: ChannelRequest):
    """Increase or decrease the intensity of the Red channel."""
    img = decode_image(req.image)
    result = img.copy().astype(np.int32)
    result[:, :, 2] = np.clip(result[:, :, 2] + req.value, 0, 255)
    return image_response(result.astype(np.uint8))


@router.post("/green")
def change_green(req: ChannelRequest):
    """Increase or decrease the intensity of the Green channel."""
    img = decode_image(req.image)
    result = img.copy().astype(np.int32)
    result[:, :, 1] = np.clip(result[:, :, 1] + req.value, 0, 255)
    return image_response(result.astype(np.uint8))


@router.post("/blue")
def change_blue(req: ChannelRequest):
    """Increase or decrease the intensity of the Blue channel."""
    img = decode_image(req.image)
    result = img.copy().astype(np.int32)
    result[:, :, 0] = np.clip(result[:, :, 0] + req.value, 0, 255)
    return image_response(result.astype(np.uint8))


@router.post("/swap/red-green")
def swap_red_green(req: ImageRequest):
    """Swap the red and green channels."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, [0, 1, 2]] = result[:, :, [0, 2, 1]]
    return image_response(result)


@router.post("/swap/red-blue")
def swap_red_blue(req: ImageRequest):
    """Swap the red and blue channels."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, [0, 1, 2]] = result[:, :, [2, 1, 0]]
    return image_response(result)


@router.post("/swap/green-blue")
def swap_green_blue(req: ImageRequest):
    """Swap the green and blue channels."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, [0, 1, 2]] = result[:, :, [1, 0, 2]]
    return image_response(result)


@router.post("/eliminate/red")
def eliminate_red(req: ImageRequest):
    """Eliminate the red channel (set to 0)."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, 2] = 0
    return image_response(result)


@router.post("/eliminate/green")
def eliminate_green(req: ImageRequest):
    """Eliminate the green channel (set to 0)."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, 1] = 0
    return image_response(result)


@router.post("/eliminate/blue")
def eliminate_blue(req: ImageRequest):
    """Eliminate the blue channel (set to 0)."""
    img = decode_image(req.image)
    result = img.copy()
    result[:, :, 0] = 0
    return image_response(result)
