"""Dialogue engine: input validation -> NLU -> response selection.

The engine has no web or UI code, so the same class serves the Gradio web
app (app.py), the command-line client (cli.py) and the unit tests.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import skills
from .nlu import IntentResult, NLUProvider, build_default_provider

INTENTS_PATH = Path(__file__).with_name("intents.json")

MAX_LEN = 500
CONFIDENT = 0.45   # at or above: answer directly
CLARIFY = 0.25     # between CLARIFY and CONFIDENT: ask "did you mean ...?"

FRIENDLY = {
    "greeting": "say hello", "how_are_you": "ask how I'm doing",
    "capabilities": "see what I can do", "bot_identity": "know who I am",
    "how_do_you_work": "know how I work", "set_name": "tell me your name",
    "get_name": "ask what your name is", "time": "know the time",
    "date": "know today's date", "math": "do a calculation",
    "joke": "hear a joke", "echo": "have me echo something",
    "reverse": "reverse some text", "chatbot_lifecycle": "learn the chatbot development lifecycle",
    "traditional_vs_llm": "compare traditional bots and LLMs",
    "thanks": "say thanks", "goodbye": "say goodbye",
}

YES = re.compile(r"^(y|ya|yes|yeah|yep|sure|ok|okay|correct|right)\b", re.I)
NO = re.compile(r"^(n|no|nope|nah|not really|wrong)\b", re.I)


@dataclass
class Reply:
    text: str
    intent: str | None
    confidence: float
    source: str


class ChatBot:
    def __init__(self, intents_path: Path = INTENTS_PATH, nlu: NLUProvider | None = None, seed: int | None = None):
        data = json.loads(Path(intents_path).read_text(encoding="utf-8"))
        self.intents = {i["tag"]: i for i in data["intents"]}
        self.nlu = nlu or build_default_provider(data["intents"])
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------ input
    @staticmethod
    def clean(raw) -> tuple[str | None, str | None]:
        """Return (clean_text, problem). Exactly one of them is None."""
        if raw is None:
            return None, "empty"
        if not isinstance(raw, str):
            raw = str(raw)
        text = unicodedata.normalize("NFKC", raw)
        text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch == " ")
        text = re.sub(r"<[^>]+>", " ", text)          # strip HTML/markup tags
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None, "empty"
        if len(text) > MAX_LEN:
            return None, "too_long"
        if not re.search(r"[A-Za-z0-9]", text):
            return None, "no_words"
        text = re.sub(r"([^\d])\1{2,}", r"\1", text)  # "heyyyy!!!" -> "hey!" (digits kept)
        letters = re.findall(r"[A-Za-z]+", text)
        if letters and not re.search(r"\d", text):
            long_words = [w for w in letters if len(w) >= 5]
            no_vowel = [w for w in long_words if not re.search(r"[aeiouy]", w, re.I)]
            if long_words and len(no_vowel) / len(long_words) > 0.5:
                return None, "gibberish"
        return text, None

    MALFORMED = {
        "empty": "It looks like your message was empty. Type something, or 'help' to see what I can do.",
        "too_long": f"That message is too long for me (limit {MAX_LEN} characters). Could you shorten it?",
        "no_words": "I can only read letters and numbers, and that message didn't have any. Try typing a question.",
        "gibberish": "Hmm, that doesn't look like words I know. Could you rephrase? Type 'help' for examples.",
    }

    # ---------------------------------------------------------------- respond
    def respond(self, raw, session: dict | None = None) -> Reply:
        session = session if session is not None else {}
        session.setdefault("misses", 0)
        session.setdefault("log", [])

        text, problem = self.clean(raw)
        if problem:
            reply = Reply(self.MALFORMED[problem], None, 0.0, f"validator:{problem}")
            return self._log(session, raw, reply)

        # Follow-up to a "did you mean ...?" question
        pending = session.pop("pending", None)
        if pending:
            if YES.match(text):
                return self._log(session, raw, self._answer(pending["intent"], pending["text"], session, 1.0, "clarified"))
            if NO.match(text):
                reply = Reply("OK, my mistake. Could you say it another way? Type 'help' to see examples.", None, 0.0, "clarified")
                return self._log(session, raw, reply)

        result: IntentResult = self.nlu.predict(text)

        if result.tag and result.confidence >= CONFIDENT:
            session["misses"] = 0
            return self._log(session, raw, self._answer(result.tag, text, session, result.confidence, result.source))

        if result.tag and result.confidence >= CLARIFY:
            session["pending"] = {"intent": result.tag, "text": text}
            guess = FRIENDLY.get(result.tag, result.tag)
            reply = Reply(f"I'm not completely sure. Did you want to {guess}? (yes/no)", result.tag, result.confidence, result.source)
            return self._log(session, raw, reply)

        session["misses"] += 1
        if session["misses"] >= 2:
            session["misses"] = 0
            msg = "I'm still not following, sorry. " + skills.capabilities(text, session)
        else:
            msg = self.rng.choice([
                "Sorry, I didn't understand that. Could you rephrase? Type 'help' to see what I can do.",
                "I'm not sure what you mean. I'm a simple rule-based bot; type 'help' for things I understand.",
            ])
        return self._log(session, raw, Reply(msg, None, result.confidence, "fallback"))

    def _answer(self, tag: str, text: str, session: dict, conf: float, source: str) -> Reply:
        intent = self.intents.get(tag, {})
        handler = skills.HANDLERS.get(intent.get("handler", ""))
        if handler:
            body = handler(text, session)
        elif intent.get("responses"):
            body = self.rng.choice(intent["responses"])
        else:
            body = "I know what you mean, but I don't have an answer for that yet."
        if tag == "greeting" and session.get("name"):
            body = body.replace("Hello!", f"Hello, {session['name']}!").replace("Hi there!", f"Hi {session['name']}!")
        return Reply(body, tag, conf, source)

    @staticmethod
    def _log(session: dict, raw, reply: Reply) -> Reply:
        # Kept in memory per session; the "monitor and improve" step of the
        # lifecycle uses it to find unmatched messages worth adding as examples.
        session["log"].append({"user": raw, "intent": reply.intent,
                               "confidence": round(reply.confidence, 3), "source": reply.source})
        return reply
