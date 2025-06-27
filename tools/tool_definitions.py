from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return weather info for a city."""
    return f"The weather in {city} is sunny and 28°C."

@tool
def get_attractions(city: str) -> str:
    """Return popular attractions for a city."""
    return f"Top attractions in {city}: Central Park, City Museum, Art Gallery."

