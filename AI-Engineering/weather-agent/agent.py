"""A small LangChain agent for current weather questions."""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool(return_direct=True)
def get_current_weather(city: str) -> str:
    """Get the current weather for a city; include a country code for ambiguous names."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        return "Set OPENWEATHER_API_KEY in .env first."
    if not city.strip():
        return "Please provide a city name."

    stage = "geocoding"
    try:
        response = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": city, "limit": 1, "appid": key},
            timeout=10,
        )
        response.raise_for_status()
        locations = response.json()
        if not locations:
            return f"No location found for {city}. Try adding a country code."
        location = locations[0]

        stage = "current weather"
        response = requests.get(
            "https://api.openweathermap.org/data/4.0/onecall/current",
            params={"lat": location["lat"], "lon": location["lon"], "units": "metric", "appid": key},
            timeout=10,
        )
        response.raise_for_status()
        weather = response.json()["data"][0]
        return (
            f"Current weather in {location['name']}, {location['country']}: "
            f"{weather['temp']}°C, {weather['weather'][0]['description']}; "
            f"feels like {weather['feels_like']}°C, humidity {weather['humidity']}%."
        )
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        if status == 401 and stage == "geocoding":
            return "OpenWeather rejected the API key (HTTP 401). Check that it is valid and active."
        if status == 401:
            return "OpenWeather rejected the current weather request (HTTP 401). Check One Call 4 access."
        return f"OpenWeather {stage} request failed (HTTP {status})."
    except requests.RequestException:
        return "Could not reach OpenWeather."
    except (KeyError, IndexError, TypeError, ValueError):
        return "OpenWeather returned unexpected data."


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python agent.py "What is the weather in Mumbai, IN?"')
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("OPENWEATHER_API_KEY"):
        raise SystemExit("Set OPENWEATHER_API_KEY in weather-agent/.env first.")

    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx"), temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer current weather questions using the weather tool. Never invent weather data."),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = AgentExecutor(
        agent=create_tool_calling_agent(llm, [get_current_weather], prompt),
        tools=[get_current_weather],
        max_iterations=3,
    )
    print(agent.invoke({"input": " ".join(sys.argv[1:])})["output"])


if __name__ == "__main__":
    main()
