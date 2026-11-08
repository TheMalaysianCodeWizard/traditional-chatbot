# TradBot — a Simple Traditional Chatbot

A prototype chatbot built with **traditional, non-LLM techniques**: input
validation, regular-expression rules, and TF-IDF cosine similarity against a
small JSON knowledge base. The reference engine is Python; a JavaScript port
of the same engine runs in the browser on a free **Static** Hugging Face Space.

## What it does

| Requirement | How TradBot meets it |
|---|---|
| Responds to multiple prompts | 17 intents: greetings, small talk, identity, name memory, time, date, arithmetic, jokes, echo/reverse, chatbot concepts, thanks, goodbye |
| Lists its capabilities | `help`, `menu`, or "what can you do" |
| Handles malformed input | Empty, whitespace-only, symbol-only, over-length (>500 chars), keyboard-mash, HTML/markup, typos, bad arithmetic, divide-by-zero, huge exponents |
| Ready for extension to an AI service | Any `NLUProvider` can be plugged in; `AzureCLUProvider` activates when Azure AI Language credentials are set |

## How a message is processed

```
user text
  -> validate & clean (engine.clean)          -> malformed? polite error
  -> RegexProvider (exact rules)              -> hit? answer (confidence 1.0)
  -> TfidfProvider (word + char n-grams)      -> score >= 0.45? answer
                                              -> 0.25-0.45? "Did you mean...?" (yes/no)
                                              -> < 0.25? fallback; 2 misses in a row -> show help
  -> handler (skills.py) or canned response (intents.json)
```

## Deploy to Hugging Face (free Static Space)

Gradio and Docker Spaces now need a paid plan, so the public demo is a Static
Space. Create a Space with **SDK: Static**, then upload the four files in
`hf_space/` (`index.html`, `bot.js`, `intents.js`, `README.md`) to the root of
the Space. After editing `chatbot/intents.json`, run
`python tools/build_static.py` to refresh `hf_space/intents.js`.

`app.py` is a Gradio version for accounts with a paid plan.

## Project layout

```
hf_space/            Static Space: index.html, bot.js (JS port), intents.js, README.md
tools/build_static.py Regenerates hf_space/intents.js from chatbot/intents.json
app.py               Gradio UI (needs a paid Hugging Face plan)
cli.py               Terminal client (python cli.py --debug)
web_preview.py       Browser preview using only the standard library
chatbot/engine.py    Validation, dialogue state, response selection
chatbot/nlu.py       NLU providers: Regex, TF-IDF, Cascade, Azure CLU
chatbot/skills.py    Computed answers: help, name, time, date, math, echo, reverse
chatbot/intents.json Knowledge base: patterns, examples, responses
tests/test_engine.py 12 unit tests (Python)
tests/parity_test.js Checks the JS port against the Python engine
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt gradio
python cli.py --debug        # terminal chat
python app.py                # Gradio web UI at http://127.0.0.1:7860
python web_preview.py        # no-Gradio browser preview at http://127.0.0.1:8000
python -m unittest discover -s tests -v
python tests/dump_python_results.py > py.json && node tests/parity_test.js py.json
```

Open `hf_space/index.html` directly in a browser to try the static version.

## Extend it

* **Add an intent:** add an object to `chatbot/intents.json` with `tag`,
  `patterns`, `examples`, and `responses` (or a `handler` in `skills.py`).
* **Use a cloud AI service:** set `AZURE_LANGUAGE_ENDPOINT`,
  `AZURE_LANGUAGE_KEY`, `AZURE_CLU_PROJECT`, `AZURE_CLU_DEPLOYMENT` and
  `pip install azure-ai-language-conversations`. This applies to the
  Python/Gradio version; a browser-only page should not hold secret keys. Or subclass `NLUProvider` for any other service.
