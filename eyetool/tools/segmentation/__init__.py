def segment_image(image_base64: str, model_path: str = "default_segmentation_model.pth") -> list:
    """Segment an image using a specific model."""
    # Dummy segmentation result
    return [{"class": "object1", "mask": "base64_encoded_mask"}]
