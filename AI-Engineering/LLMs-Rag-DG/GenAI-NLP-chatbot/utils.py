import requests

OPENWEATHER_API_KEY = "your-api-key"  # Replace with your API key
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
GEOCODING_URl = "http://api.openweathermap.org/geo/1.0/direct"


def get_gecoding(location):
    params = {
        "q": location,
        "appid": OPENWEATHER_API_KEY,
    }
    response = requests.get(GEOCODING_URl, params=params)
    if response.status_code == 200:
        geocoding_data = response.json()
        print(geocoding_data)
        lat = geocoding_data[0]["lat"]
        lon = geocoding_data[0]["lon"]
        return lat, lon
    else:
        return None, None


def get_weather(location):
    gecoding = get_gecoding(location)
    print(gecoding)
    params = {
        "lat": gecoding[0],
        "lon": gecoding[1],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"  # aGet temperature in Celsius
    }
    response = requests.get(OPENWEATHER_URL, params=params)
    if response.status_code == 200:
        weather_data = response.json()
        weather_description = weather_data["weather"][0]["description"]
        temp = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        return f"The current weather in {location} is {weather_description} with a temperature of {temp}°C and humidity of {humidity}%."
    else:
        return f"Unable to fetch weather data for {location}."
