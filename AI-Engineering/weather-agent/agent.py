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
    if not city.strip():
        return "Please provide a city name."

    stage = "geocoding"
    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        response.raise_for_status()
        locations = response.json().get("results", [])
        if not locations:
            return f"No location found for {city}. Try adding a country code."
        location = locations[0]

        stage = "current weather"
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m",
            },
            timeout=10,
        )
        response.raise_for_status()
        weather = response.json()["current"]
        return (
            f"Current weather in {location['name']}, {location['country_code']}: "
            f"{weather['temperature_2m']}°C; feels like {weather['apparent_temperature']}°C, "
            f"humidity {weather['relative_humidity_2m']}%, "
            f"wind {weather['wind_speed_10m']} km/h. Source: Open-Meteo."
        )
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        return f"Open-Meteo {stage} request failed (HTTP {status})."
    except requests.RequestException:
        return "Could not reach Open-Meteo."
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return "Open-Meteo returned unexpected data."


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python agent.py "What is the weather in Mumbai, IN?"')
    load_dotenv(Path(__file__).with_name(".env"))

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
