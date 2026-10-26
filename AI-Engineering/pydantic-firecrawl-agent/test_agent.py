"""Tests for the Firecrawl search tool and PydanticAI agent wiring."""

import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import requests
from pydantic_ai.models.test import TestModel


class Response:
    def __init__(self, data, status=200):
        self.data = data
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self.data


SEARCH_RESPONSE = {
    "success": True,
    "data": {
        "web": [
            {
                "title": "Pydantic AI documentation",
                "url": "https://pydantic.dev/docs/ai/",
                "description": "Build type-safe AI agents.",
            }
        ]
    },
}


class FirecrawlAgentTests(unittest.TestCase):
    def test_missing_key_stops_before_request(self):
        from agent import search_web

        with patch.dict(os.environ, {}, clear=True), patch("agent.requests.post") as post:
            result = search_web("Pydantic AI docs")
        self.assertIn("FIRECRAWL_API_KEY", result)
        post.assert_not_called()

    def test_cli_missing_key_stops_before_model(self):
        from agent import main

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["agent.py", "Find Pydantic AI docs"]),
            patch("agent.load_dotenv"),
            patch("agent.build_agent") as build,
        ):
            with self.assertRaisesRegex(SystemExit, "FIRECRAWL_API_KEY"):
                main()
        build.assert_not_called()

    def test_chat_reuses_history_and_exits(self):
        from agent import main

        class Result:
            def __init__(self, output, messages):
                self.output = output
                self.messages = messages

            def all_messages(self):
                return self.messages

        first_messages = ["first user", "first answer"]
        second_messages = first_messages + ["second user", "second answer"]
        output = StringIO()
        with (
            patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}),
            patch.object(sys, "argv", ["agent.py"]),
            patch("agent.load_dotenv"),
            patch("agent.build_agent") as build,
            patch("builtins.input", side_effect=["First question", "Follow-up", "quit"]),
            redirect_stdout(output),
        ):
            build.return_value.run_sync.side_effect = [
                Result("First answer", first_messages),
                Result("Second answer", second_messages),
            ]
            main()

        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.return_value.run_sync.call_args_list[0].args, ("First question",))
        self.assertEqual(build.return_value.run_sync.call_args_list[1].args, ("Follow-up",))
        self.assertEqual(build.return_value.run_sync.call_args_list[1].kwargs["message_history"], first_messages)
        self.assertIn("Chatbot: Second answer", output.getvalue())

    def test_search_calls_firecrawl_with_key_and_returns_sources(self):
        from agent import search_web

        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}), patch("agent.requests.post", return_value=Response(SEARCH_RESPONSE)) as post:
            result = search_web("Pydantic AI docs")

        self.assertIn("Pydantic AI documentation", result)
        self.assertIn("https://pydantic.dev/docs/ai/", result)
        self.assertIn("Build type-safe AI agents", result)
        self.assertEqual(post.call_args.args[0], "https://api.firecrawl.dev/v2/search")
        self.assertEqual(post.call_args.kwargs["json"]["query"], "Pydantic AI docs")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer test-key")

    def test_empty_results_are_explained(self):
        from agent import search_web

        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}), patch("agent.requests.post", return_value=Response({"success": True, "data": {"web": []}})):
            result = search_web("nothing found")
        self.assertIn("No results", result)

    def test_http_error_does_not_expose_key(self):
        from agent import search_web

        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "secret-test-key"}), patch("agent.requests.post", return_value=Response({}, 401)):
            result = search_web("example")
        self.assertIn("HTTP 401", result)
        self.assertNotIn("secret-test-key", result)

    def test_pydantic_agent_invokes_search_tool(self):
        from agent import build_agent

        agent = build_agent()
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"}), patch("agent.requests.post", return_value=Response(SEARCH_RESPONSE)) as post:
            with agent.override(model=TestModel(call_tools=["search_web"])):
                result = agent.run_sync("Find Pydantic AI docs")
        self.assertTrue(post.called)
        self.assertIn("https://pydantic.dev/docs/ai/", result.output)


if __name__ == "__main__":
    unittest.main()
