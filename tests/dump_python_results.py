"""Run a fixed set of conversations through the Python engine and print the
results as JSON, for tests/parity_test.js to compare against the JS port."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chatbot import ChatBot  # noqa: E402

CONVERSATIONS = [
    ["hi", "help", "my name is ming", "what's my name", "hello", "what time is it", "what's the date"],
    ["calculate (3+4)*2", "what is 2^10", "5 times 3", "10/0", "calc 2++", "calc 9^999999", "calc 7 mod 3",
     "what is 1000000 * 3", "calc 10 divided by 4", "calculate -2^2", "12345"],
    ["tell me a jokke", "echo hello world", "reverse hello", "echo", "whats the chatbot lifecycle",
     "how are you", "are you like chatgpt", "what is tfidf", "explain your algorithm", "thx", "bye"],
    ["", "   ", "!!!???", "x" * 600, "<b>hii</b>", "heyyyyy", "helo", "qwrtzp sdfghjk", "<script>hi</script>"],
    ["purple elephant banana", "blue giraffe sandwich", "can you tell the tme", "i want to hear something funny",
     "do you remember my name", "call me alex", "who am i", "steps to build a chatbot", "why not use gpt"],
    ["tell me something", "help me with stuff", "what can u do", "show commands", "time please",
     "what day is it", "good night", "appreciate it", "tell me about yourself", "how were you made"],
    ["joke about cats please", "yes", "jokes?", "no", "hey bot", "what is your name", "spell backwards",
     "repeat after me", "how do i use you", "phases of chatbot development"],
]

out = []
bot = ChatBot(seed=0)
for conv in CONVERSATIONS:
    session, turns = {}, []
    for msg in conv:
        r = bot.respond(msg, session)
        turns.append({"input": msg, "intent": r.intent, "source": r.source, "confidence": r.confidence, "text": r.text})
    out.append(turns)
json.dump(out, sys.stdout, indent=1)
