import emoji
import datetime
from helpers import dayify


def get_emotions():
    emotions = {
        "happy": {";)", ";-)", ":)", ":-)", ":))", "(:", "(-:", ":ճ", "։Ճ",
                  '☺️', '😊', '😇', '🙂', '🙃', '😉', '😌', '😋', '😎', '🥳', '🎊', '🎉'},
        "sad": {":,)", ":')", ":/", ":-/", ":\\", ":-\\",
                ":(", ":-(", ":((", ":,(", ":'(", "):", ")-:",
                "🥲", "😞", "😔", "😟", "😕", "🙁", "☹", "😣", "😖", "😫", "😩", "😢", "😭", "😤",
                "😠", "😡", "🤬", "😨", "😰", "😥", "😓", "😐", "😪", "🤒", "🤕"},
        "funny": {":D", ":-D", ":P", ":-P", ":դ", ":Դ",
                  '😀', '😃', '😄', '😁', '😆', '😅', '😂', '🤣', '😛', '😝', '😜', '🤪', '🤠', '💩'},
        "angry": {"-_-",
                  '🤨', '🧐', '😏', '😒', '😑', '🙄', '🖕', '🖕🏿', '🖕🏻', '🖕🏾', '🖕🏼', '🖕🏽'},
        "lovely": {"<3", ":*", ":-*",
                   '😍', '🥰', '😘', '😗', '😙', '😚', '🤩', '🤗', '❤', '🧡', '💛', '💚', '💙', '💜',
                   '🖤', '🤍', '🤎', '❤️‍🔥', '❣', '💕', '💞', '💓', '💗', '💖', '💘', '💝'},
        "surprised": {":|", ":-|",
                      '😳', '😱', '😶', '😦', '😯', '😧', '😮', '😲'}
    }

    # Armenian colon
    for i in emotions:
        emotions[i] |= {j.replace(":", "։") for j in emotions[i]}

    return emotions


def get_emojis():
    emojis = list(emoji.UNICODE_EMOJI["en"])
    emojis_emoticons = [i for j in EMOTIONS.values() for i in j if i not in emojis]
    emojis += emojis_emoticons
    return emojis


EMOTIONS = get_emotions()
EMOJIS = get_emojis()
EMOJIS_emoticons = [i for i in EMOTIONS if i not in EMOJIS]
TODAY = dayify(datetime.datetime.today())
SUFFIXES = ["յի", "ի", "յին", "ին", "յից", "ից", "ն", "ը", "ս"]
WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

OPTIONS = [
    "",
    "My activity grouped by hour/weekday/month",
    "Most/least probable words to use in a chat",
    "Compare daily usage of two or more words",
    "Longest gap in a chat",
    "Word popularity by chats",
    "Word usage per day/week/month/year",
    "First/last message times per day",
    "Conversation starters/enders per day",
    "Typical startings/endings of a conversation",
    "Screen time per day",
    "Maximum conversations held at once",
    "Distribution of media in a chat",
    "People who add me to chats the most",
    "People who disappeared at some point",
    "Special names mentioned (i.e. people I gossip about)",
    "Most notable days",
    "Daily message count over time",
    "Average message length over time",
    "Word of the day/week/month/year",
    "Average time before replying",
    "Most forwarded people",
    "People getting the most forwarded messages",
    "Statistics for a word",
    "Most questionful chat",
    "Emotions by chat",
    "Longest period of everyday chatting",
    "Compare interaction with people over time"
]