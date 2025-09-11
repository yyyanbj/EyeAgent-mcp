def detect_objects(image_base64: str, model_path: str = "default_detection_model.pth") -> list:
    """Detect objects in an image using a specific model."""
    # Dummy detection result
    return [{"class": "dog", "bbox": [10, 20, 100, 150], "confidence": 0.95}]
