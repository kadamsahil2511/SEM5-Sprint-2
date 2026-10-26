"""Checks the public weather lookup and the LangChain CLI wiring."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import requests
from langchain_core.messages import AIMessage, ToolMessage
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


class MultiToolAgentTests(unittest.TestCase):
    def test_agent_registers_weather_and_web_search_tools(self):
        from agent import main

        model = MagicMock()
        model.bind_tools.return_value = RunnableLambda(lambda _: AIMessage(content="Here is the answer."))
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["agent.py", "Find the LangChain documentation"]),
            patch("agent.ChatOllama", return_value=model),
            redirect_stdout(io.StringIO()),
        ):
            main()
        bound_tools = model.bind_tools.call_args.args[0]
        self.assertEqual({tool.name for tool in bound_tools}, {"get_current_weather", "web_search"})

    def test_web_search_returns_linked_results(self):
        from agent import web_search

        with patch("agent.DDGS") as client:
            client.return_value.text.return_value = [
                {"title": "LangChain docs", "href": "https://docs.langchain.com/", "body": "Agent documentation"}
            ]
            result = web_search.invoke({"query": "LangChain docs"})

        self.assertIn("LangChain docs", result)
        self.assertIn("https://docs.langchain.com/", result)
        self.assertIn("Agent documentation", result)
        client.return_value.text.assert_called_once_with("LangChain docs", max_results=3, backend="duckduckgo")

    def test_search_answer_does_not_get_weather_credit(self):
        from agent import main

        replies = iter([
            AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": "LangChain docs"}, "id": "call_1"}]),
            AIMessage(content="See the [LangChain docs](https://docs.langchain.com/)."),
        ])
        model = MagicMock()
        model.bind_tools.return_value = RunnableLambda(lambda _: next(replies))
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["agent.py", "Find the LangChain docs"]),
            patch("agent.ChatOllama", return_value=model),
            patch("agent.DDGS", create=True) as client,
            redirect_stdout(output),
        ):
            client.return_value.text.return_value = [{"title": "LangChain docs", "href": "https://docs.langchain.com/", "body": "Documentation"}]
            main()
        self.assertEqual(output.getvalue().strip(), "See the [LangChain docs](https://docs.langchain.com/).")

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

    def test_ambiguous_country_code_retries_explicit_filter(self):
        from agent import get_current_weather

        responses = [Response({}), *weather_responses()]
        with patch("agent.requests.get", side_effect=responses) as request:
            result = get_current_weather.invoke({"city": "Mumbai, IN"})

        self.assertIn("Mumbai, IN: 30°C", result)
        self.assertEqual(request.call_args_list[1].kwargs["params"], {"name": "Mumbai", "count": 1, "countryCode": "IN"})

    def test_http_error_identifies_provider(self):
        from agent import get_current_weather

        with patch("agent.requests.get", return_value=Response({}, 503)):
            result = get_current_weather.invoke({"city": "Mumbai"})
        self.assertIn("Open-Meteo", result)
        self.assertIn("HTTP 503", result)

    def test_cli_asks_model_to_answer_after_weather_lookup(self):
        from agent import main

        prompts = []

        def respond(prompt):
            prompts.append(prompt.to_messages())
            if len(prompts) == 1:
                return AIMessage(content="", tool_calls=[{"name": "get_current_weather", "args": {"city": "Mumbai, IN"}, "id": "call_1"}])
            return AIMessage(content="It's 30°C in Mumbai right now and feels like 33°C.")

        model = MagicMock()
        model.bind_tools.return_value = RunnableLambda(respond)
        output = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["agent.py", "What is the weather in Mumbai, IN?"]),
            patch("agent.ChatOllama", return_value=model),
            patch("agent.requests.get", side_effect=weather_responses()),
            redirect_stdout(output),
        ):
            main()
        self.assertEqual(output.getvalue().strip(), "It's 30°C in Mumbai right now and feels like 33°C. Source: Open-Meteo.")
        self.assertEqual(len(prompts), 2)
        self.assertTrue(any(isinstance(message, ToolMessage) and "30°C" in message.content for message in prompts[1]))


if __name__ == "__main__":
    unittest.main()
