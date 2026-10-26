"""The full-screen chat and its slash commands."""

import asyncio
import tempfile
import unittest
from pathlib import Path

from chat_store import ChatStore
from textual.widgets import Input, RichLog
from tui import ChatApp


class FakeAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"output": "**Hello** [docs](https://example.com)", "intermediate_steps": []}


class ChatAppTests(unittest.TestCase):
    def test_full_screen_chat_renders_markdown_and_saves_turn(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                store = ChatStore(Path(directory) / "sessions.json")
                app = ChatApp(FakeAgent(), store)
                async with app.run_test() as pilot:
                    self.assertIsNotNone(app.query_one("#chat", RichLog))
                    self.assertIsNotNone(app.query_one("#question", Input))
                    await pilot.press(*"Hi", "enter")
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                    rendered = "\n".join(line.text for line in app.query_one("#chat", RichLog).lines)
                    self.assertIn("Hello", rendered)
                    self.assertIn("https://example.com", rendered)
                    self.assertNotIn("**Hello**", rendered)
                self.assertEqual(store.current()["messages"][0], {"role": "user", "content": "Hi"})
                self.assertEqual(store.current()["messages"][1]["role"], "assistant")

        asyncio.run(scenario())

    def test_new_and_load_commands_switch_persistent_sessions(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as directory:
                store = ChatStore(Path(directory) / "sessions.json")
                first_id = store.current()["id"]
                app = ChatApp(FakeAgent(), store)
                async with app.run_test() as pilot:
                    await pilot.press(*"/new", "enter")
                    await pilot.pause()
                    self.assertNotEqual(store.current()["id"], first_id)
                    await pilot.press(*f"/load {first_id}", "enter")
                    await pilot.pause()
                    self.assertEqual(store.current()["id"], first_id)
                self.assertEqual(ChatStore(store.path).current()["id"], first_id)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
