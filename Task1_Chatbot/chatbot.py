# ============================================================
#   RuleBot - Rule-Based Chatbot
#   CodSoft AI Internship | Task 1
#   Author: Your Name
# ============================================================

import re
import datetime

def get_response(user_input):
    text = user_input.lower().strip()

    # greetings
    if re.search(r'\b(hello|hi|hey|howdy|hiya|greetings)\b', text, re.IGNORECASE):
        return "Hey! What's up? 👋"

    # asking how the bot is doing
    elif re.search(r'\b(how are you|how r u|how do you do|whats up|what\'s up|you good)\b', text, re.IGNORECASE):
        return "Honestly? Pretty good for a script. 😄 You?"

    # asking who/what the bot is
    elif re.search(r'\b(who are you|your name|what are you|introduce yourself)\b', text, re.IGNORECASE):
        return "I'm RuleBot — nothing fancy, just a chatbot made for the CodSoft AI internship. 🤖"

    # age question
    elif re.search(r'\b(how old|your age|age)\b', text, re.IGNORECASE):
        return "I came to life the second you ran this script, so... very young. 😂"

    # AI and ML topics
    elif re.search(r'\b(artificial intelligence|machine learning|deep learning|ai|ml|neural)\b', text, re.IGNORECASE):
        return "AI is basically teaching machines to think like humans. ML takes it further — instead of telling the machine what to do, you show it data and let it figure things out on its own. Pretty wild honestly. 🧠"

    # programming and python
    elif re.search(r'\b(python|coding|programming|code|developer)\b', text, re.IGNORECASE):
        return "Python's genuinely a great pick for AI stuff. Libraries like NumPy, Pandas, and TensorFlow do a lot of the heavy lifting. 🐍"

    # codsoft and internship
    elif re.search(r'\b(codsoft|internship|task|certificate)\b', text, re.IGNORECASE):
        return "CodSoft internship's pretty solid for building actual projects. Finish at least 3 tasks and you get the certificate. Worth it! 📜"

    # jokes
    elif re.search(r'\b(joke|funny|laugh|humor|hilarious)\b', text, re.IGNORECASE):
        return "Alright —\nWhy did the programmer quit his job?\nBecause he didn't get arrays. 😂\n(I'll see myself out)"

    # current time
    elif re.search(r'\b(time|clock|what time)\b', text, re.IGNORECASE):
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"It's {now} right now. ⏰"

    # current date
    elif re.search(r'\b(date|today|day|what day)\b', text, re.IGNORECASE):
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        return f"Today's {today}. 📅"

    # weather
    elif re.search(r'\b(weather|temperature|rain|sunny|forecast)\b', text, re.IGNORECASE):
        return "No weather API here unfortunately! But hoping it's nice wherever you are. ☀️"

    # help menu
    elif re.search(r'\b(help|what can you do|commands|options)\b', text, re.IGNORECASE):
        return (
            "Here's what I can actually help with:\n"
            "  • Saying hi and small talk\n"
            "  • AI and Python questions\n"
            "  • Cracking a (bad) joke\n"
            "  • Telling you the time or date\n"
            "  • CodSoft internship stuff\n\n"
            "Just type whatever, I'll do my best! 😊"
        )

    # thank you
    elif re.search(r'\b(thank|thanks|ty|thx|thank you)\b', text, re.IGNORECASE):
        return "No worries at all! 😊"

    # who made the bot
    elif re.search(r'\b(who made you|who created you|who built you|your creator)\b', text, re.IGNORECASE):
        return "A CodSoft intern built me for Task 1. Simple but gets the job done! 💻"

    # goodbye
    elif re.search(r'\b(bye|goodbye|exit|quit|see you|cya|later)\b', text, re.IGNORECASE):
        return "EXIT"

    # empty input
    elif text == "":
        return "You there? Try typing something — even just 'hi' works. 😊"

    # fallback for unknown input
    else:
        return "Not sure how to answer that one. 🤔 Try 'help' to see what I can do."


def print_banner():
    print("=" * 50)
    print("         RuleBot — CodSoft AI Internship")
    print("              Task 1 : Rule-Based Chatbot")
    print("=" * 50)
    print("  say 'help' to see topics  |  'bye' to quit")
    print("=" * 50)
    print()


def main():
    print_banner()
    print("Bot: Hey! I'm RuleBot. Ask me anything. 🤖\n")

    while True:
        try:
            user_input = input("You: ").strip()
            response = get_response(user_input)

            if response == "EXIT":
                print("Bot: Later! 👋 Hope that was useful.")
                print()
                break

            print(f"Bot: {response}")
            print()

        except KeyboardInterrupt:
            print("\nBot: Oh — Ctrl+C. Fair enough. Bye! 👋")
            break


if __name__ == "__main__":
    main()