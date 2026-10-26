import os
import io
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


class WeatherToolTests(unittest.TestCase):
    def test_cli_runs_native_weather_tool_call(self):
        from agent import main

        responses = [
            Response([{"name": "Mumbai", "country": "IN", "lat": 19.076, "lon": 72.878}]),
            Response({"data": [{"temp": 30, "feels_like": 33, "humidity": 75, "weather": [{"description": "haze"}]}]}),
        ]
        model = MagicMock()
        model.bind_tools.return_value = RunnableLambda(
            lambda _: AIMessage(content="", tool_calls=[{"name": "get_current_weather", "args": {"city": "Mumbai,IN"}, "id": "call_1"}])
        )
        output = io.StringIO()
        with (
            patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}),
            patch.object(sys, "argv", ["agent.py", "What is the weather in Mumbai, IN?"]),
            patch("agent.ChatOllama", return_value=model, create=True),
            patch("agent.requests.get", side_effect=responses),
            redirect_stdout(output),
        ):
            main()
        self.assertIn("Current weather in Mumbai, IN: 30°C", output.getvalue())

    def test_city_lookup_uses_one_call_4_current_weather(self):
        from agent import get_current_weather

        responses = [
            Response([{"name": "Mumbai", "country": "IN", "lat": 19.076, "lon": 72.878}]),
            Response({"data": [{"temp": 30, "feels_like": 33, "humidity": 75, "weather": [{"description": "haze"}]}]}),
        ]
        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}), patch("agent.requests.get", side_effect=responses) as request:
            result = get_current_weather.invoke({"city": "Mumbai,IN"})

        self.assertIn("Mumbai, IN", result)
        self.assertIn("30°C", result)
        self.assertIn("haze", result)
        self.assertNotIn("test-key", result)
        self.assertEqual(request.call_args_list[0].args[0], "https://api.openweathermap.org/geo/1.0/direct")
        self.assertEqual(request.call_args_list[1].args[0], "https://api.openweathermap.org/data/4.0/onecall/current")
        self.assertEqual(request.call_args_list[1].kwargs["params"]["units"], "metric")

    def test_missing_key_stops_before_request(self):
        from agent import get_current_weather

        with patch.dict(os.environ, {}, clear=True), patch("agent.requests.get") as request:
            result = get_current_weather.invoke({"city": "Mumbai"})
        self.assertIn("OPENWEATHER_API_KEY", result)
        request.assert_not_called()

    def test_http_error_does_not_expose_key(self):
        from agent import get_current_weather

        with patch.dict(os.environ, {"OPENWEATHER_API_KEY": "test-key"}), patch("agent.requests.get", return_value=Response({}, 401)):
            result = get_current_weather.invoke({"city": "Mumbai"})
        self.assertIn("HTTP 401", result)
        self.assertIn("API key", result)
        self.assertNotIn("subscription", result)
        self.assertNotIn("test-key", result)


if __name__ == "__main__":
    unittest.main()
