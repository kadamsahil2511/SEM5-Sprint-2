"""Checks the public weather lookup and the LangChain CLI wiring."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.data


def weather_responses():
    return [
        Response({"results": [{"name": "Mumbai", "country_code": "IN", "latitude": 19.076, "longitude": 72.878}]}),
        Response({"current": {"temperature_2m": 30, "apparent_temperature": 33, "relative_humidity_2m": 75, "wind_speed_10m": 12}}),
    ]


class WeatherToolTests(unittest.TestCase):
    def test_weather_lookup_needs_no_api_key(self):
        from agent import get_current_weather

        with patch.dict(os.environ, {}, clear=True), patch("agent.requests.get", side_effect=weather_responses()) as request:
            result = get_current_weather.invoke({"city": "Mumbai, IN"})

        self.assertIn("Mumbai, IN: 30°C", result)
        self.assertIn("feels like 33°C", result)
        self.assertIn("humidity 75%", result)
        self.assertIn("Open-Meteo", result)
        self.assertEqual(request.call_args_list[0].args[0], "https://geocoding-api.open-meteo.com/v1/search")
        self.assertEqual(request.call_args_list[0].kwargs["params"], {"name": "Mumbai, IN", "count": 1})
        self.assertEqual(request.call_args_list[1].args[0], "https://api.open-meteo.com/v1/forecast")
        self.assertNotIn("apikey", request.call_args_list[1].kwargs["params"])

    def test_unknown_city_stops_before_forecast(self):
        from agent import get_current_weather

        with patch("agent.requests.get", return_value=Response({})) as request:
            result = get_current_weather.invoke({"city": "NoSuchCity"})
        self.assertIn("No location found", result)
        request.assert_called_once()

    def test_http_error_identifies_provider(self):
        from agent import get_current_weather

        with patch("agent.requests.get", return_value=Response({}, 503)):
            result = get_current_weather.invoke({"city": "Mumbai"})
        self.assertIn("Open-Meteo", result)
        self.assertIn("HTTP 503", result)

    def test_cli_runs_without_weather_key(self):
        from agent import main

        model = MagicMock()
        model.bind_tools.return_value = RunnableLambda(
            lambda _: AIMessage(content="", tool_calls=[{"name": "get_current_weather", "args": {"city": "Mumbai, IN"}, "id": "call_1"}])
        )
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["agent.py", "What is the weather in Mumbai, IN?"]),
            patch("agent.ChatOllama", return_value=model),
            patch("agent.requests.get", side_effect=weather_responses()),
            redirect_stdout(output),
        ):
            main()
        self.assertIn("Current weather in Mumbai, IN: 30°C", output.getvalue())


if __name__ == "__main__":
    unittest.main()
