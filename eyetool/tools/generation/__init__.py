def generate_image(prompt: str, model_path: str = "default_generation_model.pth") -> str:
    """Generate an image from a text prompt using a specific model."""
    # Dummy generated image
    return "base64_encoded_generated_image"

def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

def sum_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Return the product of two numbers."""
    return a * b
