"""Gradio web front end - the entry point Hugging Face Spaces runs.

The engine keeps per-conversation state (your name, pending "did you mean"
questions). Rather than store that on the server, each request rebuilds it
by replaying the user's earlier messages from the chat history. That keeps
the Space stateless, so many visitors can chat at once without mixing up
their sessions.
"""

import gradio as gr

from chatbot import ChatBot

bot = ChatBot()

EXAMPLES = ["help", "my name is Sam", "what time is it", "calculate (3 + 4) * 2",
            "tell me a joke", "what is the chatbot development lifecycle", "asdfgh qwrtzp"]


def _user_turns(history):
    for turn in history or []:
        if isinstance(turn, dict):                       # "messages" format
            if turn.get("role") == "user" and isinstance(turn.get("content"), str):
                yield turn["content"]
        elif isinstance(turn, (list, tuple)) and turn:    # legacy tuple format
            yield turn[0]


def chat(message, history):
    session: dict = {}
    for previous in _user_turns(history):
        bot.respond(previous, session)
    return bot.respond(message, session).text


DESCRIPTION = (
    "A **traditional (non-LLM) chatbot**. Messages are validated, matched against "
    "hand-written regex rules, then against example phrases with TF-IDF cosine similarity. "
    "Low-confidence matches trigger a clarifying question; unmatched input gets a fallback. "
    "Type **help** to list capabilities."
)

_kwargs = dict(fn=chat, title="TradBot - a Traditional Chatbot", description=DESCRIPTION, examples=EXAMPLES)
try:
    demo = gr.ChatInterface(type="messages", **_kwargs)   # Gradio 4.4x / 5.x
except TypeError:
    demo = gr.ChatInterface(**_kwargs)                    # newer Gradio: messages is the default

if __name__ == "__main__":
    demo.launch()
