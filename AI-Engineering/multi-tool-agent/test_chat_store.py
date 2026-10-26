"""Persistent session behavior for the terminal chat."""

import tempfile
import unittest
from pathlib import Path

from chat_store import ChatStore


class ChatStoreTests(unittest.TestCase):
    def test_turns_and_selected_session_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.json"
            store = ChatStore(path)
            first_id = store.current()["id"]
            store.add_turn("Weather in Mumbai?", "It is warm.")
            second_id = store.new_session()["id"]
            store.add_turn("Search for LangChain", "Here are docs.")

            reopened = ChatStore(path)
            self.assertEqual(reopened.current()["id"], second_id)
            self.assertEqual(reopened.current()["messages"], [
                {"role": "user", "content": "Search for LangChain"},
                {"role": "assistant", "content": "Here are docs."},
            ])
            self.assertEqual(len(reopened.list_sessions()), 2)
            reopened.load(first_id)
            self.assertEqual(reopened.current()["messages"][0]["content"], "Weather in Mumbai?")
            self.assertEqual(ChatStore(path).current()["id"], first_id)


if __name__ == "__main__":
    unittest.main()
