import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    api_key = os.getenv("WHETHER_API_KEY")
    if not api_key:
        return "Weather API key not found. Please set 'WHETHER_API_KEY' in your .env file."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if response.status_code == 200:
            temp = data['main']['temp']
            condition = data['weather'][0]['description'].capitalize()
            return f"The weather in {city} is {condition} with a temperature of {temp}°C."
        elif response.status_code == 404:
            return f"City '{city}' not found in weather database."
        else:
            return f"Failed to fetch weather. Error: {data.get('message', 'Unknown error')}"

    except requests.exceptions.RequestException as e:
        return f"Network error while fetching weather: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
