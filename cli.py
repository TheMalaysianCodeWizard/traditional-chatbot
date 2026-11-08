"""Command-line client for testing the bot without a browser.

    python cli.py           # chat interactively
    python cli.py --debug   # also show matched intent, confidence and source
"""

import sys

from chatbot import ChatBot


def main() -> None:
    debug = "--debug" in sys.argv
    bot, session = ChatBot(), {}
    print("TradBot (type 'help' for capabilities, 'quit' to exit)")
    while True:
        try:
            msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        reply = bot.respond(msg, session)
        print(f"Bot: {reply.text}")
        if debug:
            print(f"     [intent={reply.intent} confidence={reply.confidence:.2f} source={reply.source}]")
        if reply.intent == "goodbye":
            break


if __name__ == "__main__":
    main()
