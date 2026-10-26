"""Full-screen terminal chat for the multi-tool agent."""

from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Input, RichLog, Static

from chat_store import ChatStore


def answer_question(agent, question, history):
    result = agent.invoke({"input": question, "chat_history": history})
    answer = result["output"].strip()
    if any(
        action.tool == "get_current_weather" and observation.startswith("Current weather in ")
        for action, observation in result["intermediate_steps"]
    ) and "Open-Meteo" not in answer:
        answer += " Source: Open-Meteo."
    return answer


class ChatApp(App):
    TITLE = "Multi-tool agent"
    BINDINGS = [("ctrl+q", "quit", "Quit")]
    CSS = """
    Screen { layout: vertical; }
    #chat { height: 1fr; border: round $primary; padding: 1 2; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #question { height: 3; margin: 0 1; }
    """

    def __init__(self, agent, store: ChatStore):
        super().__init__()
        self.agent = agent
        self.store = store

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="chat", wrap=True, min_width=1)
        yield Static("", id="status")
        yield Input(placeholder="Ask a question or type /help", id="question")
        yield Footer()

    def on_mount(self):
        self._show_session()
        self.query_one("#question", Input).focus()

    def _show_session(self):
        session = self.store.current()
        self.sub_title = f"Session {session['id']}"
        chat = self.query_one("#chat", RichLog)
        chat.clear()
        if not session["messages"]:
            chat.write("Ask about current weather or search the web. Type /help for commands.")
        for message in session["messages"]:
            if message["role"] == "user":
                chat.write(Text(f"You: {message['content']}", style="bold cyan"))
            else:
                chat.write(Text("Assistant", style="bold green"))
                chat.write(Markdown(message["content"], hyperlinks=False))

    def on_input_submitted(self, event: Input.Submitted):
        question = event.value.strip()
        event.input.value = ""
        if not question:
            return
        if question.startswith("/"):
            self._command(question)
            return

        self.query_one("#chat", RichLog).write(Text(f"You: {question}", style="bold cyan"))
        event.input.disabled = True
        self.query_one("#status", Static).update("Thinking...")
        self._ask(question)

    def _command(self, text):
        name, _, argument = text.partition(" ")
        chat = self.query_one("#chat", RichLog)
        if name == "/help":
            chat.write("/new  Start a new chat    /sessions  List saved chats")
            chat.write("/load ID  Open a saved chat    /quit  Exit")
        elif name == "/new":
            self.store.new_session()
            self._show_session()
        elif name == "/sessions":
            for session in self.store.list_sessions():
                chat.write(f"{session['id']}  {session['title']}")
        elif name == "/load":
            try:
                self.store.load(argument.strip())
            except ValueError as error:
                chat.write(str(error))
            else:
                self._show_session()
        elif name == "/quit":
            self.exit()
        else:
            chat.write(f"Unknown command: {name}. Type /help.")

    @work(thread=True)
    def _ask(self, question):
        history = [
            HumanMessage(content=message["content"]) if message["role"] == "user"
            else AIMessage(content=message["content"])
            for message in self.store.current()["messages"]
        ]
        try:
            answer = answer_question(self.agent, question, history)
        except Exception as error:
            self.call_from_thread(self._show_error, str(error))
        else:
            self.call_from_thread(self._show_answer, question, answer)

    def _show_answer(self, question, answer):
        self.store.add_turn(question, answer)
        chat = self.query_one("#chat", RichLog)
        chat.write(Text("Assistant", style="bold green"))
        chat.write(Markdown(answer, hyperlinks=False))
        self._enable_input()

    def _show_error(self, error):
        self.query_one("#chat", RichLog).write(f"Unable to get an answer: {error}")
        self._enable_input()

    def _enable_input(self):
        self.query_one("#status", Static).update("")
        question = self.query_one("#question", Input)
        question.disabled = False
        question.focus()
