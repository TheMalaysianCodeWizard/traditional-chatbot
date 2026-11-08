---
title: TradBot Traditional Chatbot
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: A rule-based + TF-IDF chatbot (no LLM), runs in the browser
---

# TradBot (Static Space)

A traditional, non-LLM chatbot that runs entirely in the browser. This folder
is everything the free **Static** Hugging Face Space needs:

| File | Purpose |
|---|---|
| `index.html` | Chat page |
| `bot.js` | JavaScript port of the Python engine (validation, regex + TF-IDF, skills) |
| `intents.js` | Knowledge base, generated from `chatbot/intents.json` |
| `README.md` | This file; the block at the top holds the Space settings |

Source, Python version, and tests: see the GitHub repository.
