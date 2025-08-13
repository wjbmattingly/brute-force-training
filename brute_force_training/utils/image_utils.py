"""
Image processing utilities for vision-language model training
"""

from PIL import Image
import base64
from io import BytesIO
from typing import Union


def ensure_pil_image(image: Union[Image.Image, str], min_size: int = 256, max_size: int = 500) -> Image.Image:
    """
    Ensures that the input image is a PIL Image object and meets size requirements.

    This function handles different input types:
    - If the input is already a PIL Image, it's used directly.
    - If the input is a string, it's assumed to be a base64-encoded image and is decoded.
    - For other input types, a ValueError is raised.

    The function resizes the image if it's smaller than the minimum size or larger than
    the maximum size, maintaining the aspect ratio.

    Args:
        image: The input image, either as a PIL Image object or a base64-encoded string.
        min_size: The minimum size (in pixels) for both width and height. Defaults to 256.
        max_size: The maximum size (in pixels) for the larger dimension. Defaults to 500.

    Returns:
        A PIL Image object meeting the size requirements.

    Raises:
        ValueError: If the input image type is not supported.
    """
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        # Assuming it's a base64 string
        if image.startswith('data:image'):
            image = image.split(',')[1]
        image_data = base64.b64decode(image)
        pil_image = Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Check if the image exceeds the maximum size
    if max(pil_image.width, pil_image.height) > max_size:
        # Calculate the scaling factor to fit within max_size
        scale = max_size / max(pil_image.width, pil_image.height)
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        
        # Resize the image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    # Check if the image is smaller than the minimum size
    elif pil_image.width < min_size or pil_image.height < min_size:
        # Calculate the scaling factor
        scale = max(min_size / pil_image.width, min_size / pil_image.height)
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        
        # Resize the image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    return pil_image
