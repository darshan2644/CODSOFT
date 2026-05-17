# Task 1 - Rule-Based Chatbot

A simple terminal chatbot built with pure Python. It uses pattern matching with regular expressions to understand what the user typed and responds using predefined rules. No AI or machine learning involved — just smart if-else logic.

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `re` | Pattern matching to detect keywords in user input |
| `datetime` | Fetching current time and date for time-related responses |

Both are part of Python's standard library — no installation needed.

---

## How to Run

No external packages required. Just run the script directly.

```bash
python chatbot.py
```

---

## Sample Output

```
============================================================
        Welcome to the AI Chatbot!
  Type 'quit' or 'bye' to exit the conversation.
============================================================

You: hello
Bot: Hey there! Great to meet you! How can I help you today?

You: tell me a joke
Bot: Why don't scientists trust atoms? Because they make up everything! 😄

You: what is artificial intelligence
Bot: Artificial Intelligence is the simulation of human intelligence by machines.
     It includes learning, reasoning, and self-correction. Pretty cool, right?

You: what time is it
Bot: The current time is 11:30:45 AM

You: bye
Bot: Goodbye! It was great chatting with you. Have an awesome day! 👋

============================================================
  Thanks for chatting! See you next time!
============================================================
```
