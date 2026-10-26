import streamlit as st
from langchain_community.llms.ollama import Ollama

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory

# Constants
PROMPT_TEMPLATE = """
You are good and intelligent assistant who can help with studies, answers their doubts and home automation.
You can also provide weather updates for cities.
If the user greets, respond with a greeting. If the user asks a question, provide an answer.
You can use following context too for answering questions:

{context}

Conversation History:
{history}

---


Answer the question based on the above context: {query}

"""

OLLAMA_API_URL = "http://localhost:11434"

st.set_page_config(page_title="Smart Study Buddy", layout="wide")
st.title("Smart Study Buddy")
st.write("Ask questions about study tips, tricks, or get weather updates with home automation!")

llm = Ollama(
    base_url=OLLAMA_API_URL,
    model="deepseek-r1:1.5b",  # Replace with the specific model name you are using
    timeout=10  # Add a timeout of 10 seconds
)


# Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="query",
)

# Prompt Template for LangChain
prompt_template = PromptTemplate(
    input_variables=["context", "history", "query"],
    template=PROMPT_TEMPLATE
)

# LangChain LLM Chain
chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)

st.sidebar.title("Options")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
# Chatbox implementation
st.subheader("Chatbox")


# Container for chat messages
chat_container = st.container()

# Function to display chat messages
def display_message(message, is_user=True):
    if is_user:
        chat_container.markdown(f"<div style='text-align: right; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)
    else:
        chat_container.markdown(f"<div style='text-align: left; padding: 10px; border-radius: 10px; margin: 5px;'>{message}</div>", unsafe_allow_html=True)

def send_to_llm():
    user_input = st.session_state.user_input
    try:
        user_input = user_input.strip()
        context = ""
        response = chain.run(context=context, query=user_input)
        # Update conversation memory
        st.session_state.messages.append({"role": "user", "content": user_input, "is_user": True})
        st.session_state.messages.append({"role": "assistant", "content": response, "is_user": False})
        st.rerun()
        return response
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
with chat_container:
    for chat in st.session_state.messages:
        display_message(chat['content'], is_user=chat['is_user'])


# User Input Section
user_input = st.chat_input("Enter your query:", key="user_input", on_submit=send_to_llm)
