import random

def get_weather(city: str) -> str:
    """Get fake weather information for a city."""
    weathers = ["Sunny 27°C", "Cloudy 25°C", "Rainy 23°C"]
    return f"Weather in {city}: {random.choice(weathers)}"
