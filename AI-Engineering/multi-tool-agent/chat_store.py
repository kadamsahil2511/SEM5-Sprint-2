"""Save multi-tool chat sessions on the local machine."""

import json
import os
from pathlib import Path
from uuid import uuid4


class ChatStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {"current_id": None, "sessions": []}
        if not self.data["sessions"]:
            self.new_session()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)

    def current(self):
        return next(session for session in self.data["sessions"] if session["id"] == self.data["current_id"])

    def list_sessions(self):
        return list(reversed(self.data["sessions"]))

    def new_session(self):
        session = {"id": uuid4().hex[:8], "title": "New chat", "messages": []}
        self.data["sessions"].append(session)
        self.data["current_id"] = session["id"]
        self._save()
        return session

    def load(self, session_id: str):
        session = next((item for item in self.data["sessions"] if item["id"] == session_id), None)
        if session is None:
            raise ValueError(f"No session with ID {session_id}.")
        self.data["current_id"] = session_id
        self._save()
        return session

    def add_turn(self, question: str, answer: str):
        session = self.current()
        if not session["messages"]:
            session["title"] = question[:60]
        session["messages"].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        self._save()
