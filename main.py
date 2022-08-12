import re
import json
import emoji
import datetime
import warnings
import numpy as np
import pandas as pd
import more_itertools as mit

import streamlit as st
import streamlit.components.v1 as components

import bar_chart_race as bcr
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from helpers import *
from constants import *


warnings.filterwarnings("ignore")

st.image("neplo-logo.png", width=150)
st.header('Welcome to ñeplo!')
st.subheader('Upload your Telegram data and let ñeplo do the magic.')


def get_chats_df(file):
    data = json.load(file)
    user = data["personal_information"]["first_name"]
    user += " "
    user += data["personal_information"]["last_name"]
    user = user.strip()
    chats_df = pd.json_normalize(data["chats"]["list"])
    chats_df.messages = chats_df.messages.apply(pd.DataFrame)
    for i, line in chats_df.iterrows():
        line.messages["chat_name"] = line["name"]
        line.messages["chat_id"] = line.id
        line.messages["pm"] = line.type == "personal_chat"
    return user, chats_df


def get_messages_df(chat_length_threshold=3):
    messages_df = pd.concat(chats_df.messages.tolist())
    messages_df.index = range(len(messages_df))
    messages_df.date = pd.to_datetime(messages_df.date)
    messages_df.photo = ~messages_df.photo.isna()
    messages_df.file = ~messages_df.file.isna()
    important_cols = ['id', 'type', 'date', 'date_unixtime', 'actor', 'actor_id', 'action', 'text', 'from', 'from_id',
                      'reply_to_message_id', 'photo', 'file', 'media_type', 'sticker_emoji', 'forwarded_from',
                      'chat_name', 'chat_id', 'pm']
    messages_df = messages_df[important_cols]
    chat_counts = messages_df.groupby("chat_name").apply(lambda x: ((x.type == "message") & (x["from"] == USER)).sum())
    messages_df = messages_df[messages_df.chat_name.isin(chat_counts.index[chat_counts >= chat_length_threshold])]
    messages_df["link"] = messages_df.text.apply(contains_link)
    messages_df["gif"] = messages_df.media_type == "animation"
    messages_df["voice"] = messages_df.media_type == "voice_message"
    messages_df["video"] = messages_df.media_type.isin(["video_file", "video_message"])
    messages_df.text = messages_df.text.apply(
        lambda x: " ".join([i if type(i) == str else i["text"] for i in x]) if type(x) == list else x)
    messages_df["question_mark"] = messages_df.text.str.contains("\?|՞")
    messages_df["emoji_list"] = messages_df.text.str.findall(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    messages_df.emoji_list += messages_df.text.apply(lambda x: [i for i in EMOJIS_emoticons if i in x])
    messages_df.emoji_list = messages_df.emoji_list.apply(set)
    messages_df["emoji"] = messages_df.emoji_list.apply(len).astype(bool)
    for i, j in EMOTIONS.items():
        messages_df[f"emotion_{i}"] = messages_df.emoji_list.apply(lambda x: len(x & j))
    messages_df.text = messages_df.text.str.replace(r"[։:]\-?[դԴճՃdDpP]", " ").str.replace("[՞՛՜]+", "").str.replace(
        "\W+", " ").str.strip()
    mask = ~messages_df.media_type.isna() & (messages_df.text == "")
    messages_df.type[mask] = messages_df.media_type[mask]
    messages_df["day_session"] = messages_df.date.apply(dayify)
    messages_df.date_unixtime = messages_df.date_unixtime.astype(int)
    messages_df["to"] = None
    messages_df["to"][(messages_df.type == "message") & messages_df.pm & (messages_df["from"] == USER)] = \
        messages_df[(messages_df.type == "message") & messages_df.pm & (messages_df["from"] == USER)].chat_name
    messages_df["to"][(messages_df.type == "message") & messages_df.pm & (messages_df["from"] != USER)] = USER
    messages_df["text_splitted"] = messages_df.text.str.split()
    messages_df["text_normalized"] = messages_df.text.str.lower()
    messages_df["text_normalized_splitted"] = messages_df.text_normalized.str.split()
    return messages_df


def get_word_dfs(canonical_form_threshold=10, special_name_threshold=0.95):
    global USER, messages_df
    words_df = messages_df[(messages_df.text != "") & messages_df.pm][
        ["id", "chat_name", "from", "to", "date", "day_session", "text"]]
    words_df.text = words_df.text.str.split()
    words_df = pd.DataFrame(
        [i for j in words_df.apply(lambda x: [x.tolist() + [i] for i in x.text], axis=1).values for i in j])
    del words_df[6]
    words_df.columns = ["id", "chat_name", "from", "to", "date", "day_session", "word"]
    words_df["word_normalized"] = words_df.word.str.lower()
    words_df["stopword"] = words_df.word_normalized.str.len() < 3
    with open("stopwords.txt", "r", encoding="utf-8", errors="ignore") as s:
        stopwords = s.read().split()
        words_df.stopword |= words_df.word_normalized.isin(stopwords)

    word_occurrences = words_df[words_df["from"] == USER].sort_values("date").groupby("word").day_session.apply(
        lambda x: x.values)
    words_unique_df = words_df[words_df["from"] == USER].sort_values("date").drop_duplicates(subset=["word"])
    words_unique_df = words_unique_df.query("word_normalized.str.isalpha()")
    words_unique_df.index = words_unique_df.word
    words_unique_df["birth"] = words_unique_df.word.apply(word_occurrences.apply(lambda x: x[0]).get, 1)
    words_unique_df["death"] = words_unique_df.word.apply(word_occurrences.apply(lambda x: x[-1]).get, 1)
    words_unique_df["peak_day"] = words_unique_df.word.apply(word_occurrences.apply(mod).get, 1)
    words_unique_df["lifespan"] = words_unique_df.death - words_unique_df.birth
    words_unique_df["lowercase"] = words_unique_df.word == words_unique_df.word_normalized
    words_unique_df["capital"] = words_unique_df.word == words_unique_df.word.str.capitalize()
    words_unique_df["counts"] = words_unique_df.word.apply(words_df.word.value_counts().get)
    words_unique_df["counts_normalized"] = words_unique_df.word_normalized.apply(
        words_df.word_normalized.value_counts().get)
    words_unique_df["distinct_counts"] = words_unique_df.word.apply(messages_df.text.value_counts().get).fillna(
        0).astype(
        int)
    words_unique_df["distinct_counts_normalized"] = words_unique_df.word_normalized.apply(
        messages_df.text_normalized.value_counts().get).fillna(0).astype(int)
    words_unique_df["isalpha"] = words_unique_df.word.str.isalpha()
    words_unique_df["relevant"] = words_unique_df.counts_normalized > len(
        words_df) / words_unique_df.word_normalized.nunique()
    words_df["relevant"] = words_df.word_normalized.isin(words_unique_df.query("relevant == True").word_normalized)

    words_unique_df["counts_over_normalized"] = words_unique_df.counts / words_unique_df.counts_normalized
    words_unique_df["special"] = words_unique_df.relevant & words_unique_df.capital & words_unique_df.isalpha \
                                 & (words_unique_df.counts_over_normalized >= special_name_threshold)
    special_names_df = words_unique_df[words_unique_df.special]
    special_names_df["canonical"] = special_names_df.index

    for name in special_names_df.index:
        for suf in SUFFIXES:
            if name.endswith(suf):
                canonical = name[:-len(suf)]
                if (canonical in special_names_df.index) and \
                        special_names_df.counts[canonical] >= canonical_form_threshold:
                    special_names_df.canonical[name] = canonical
                    break

    for canonical in special_names_df.canonical:
        forms = []
        canonical_ = canonical
        if canonical_[-1] not in "աեէըիուօ":
            forms.append(canonical + "ը")
        if canonical_[-1] in "աո":
            canonical_ += "յ"
        # not elegant
        forms += [canonical_ + suf.lstrip("յը") for suf in SUFFIXES]

        for form in forms:
            if (form not in special_names_df.index) and (form in words_unique_df.index):
                new_line = words_unique_df.loc[form]
                new_line["canonical"] = canonical
                special_names_df = special_names_df.append(new_line)

    return words_df, words_unique_df, special_names_df


def questions_asked_per_user(sent=True):
    col = "to" if sent else "from"
    df = messages_df[messages_df.question_mark & messages_df.pm & (messages_df[col] != USER)][col]
    counts = pd.DataFrame({"Quantity": df.value_counts(), "Percentage": df.value_counts(normalize=True)})
    return counts


def popularity_with_each_user(word, match_case=False, normalize=True):
    text = "text_splitted" if match_case else "text_normalized_splitted"
    df = messages_df[(messages_df.type == "message") & messages_df.pm][["to", text]]
    df["occurrence"] = df[text].apply(list.__contains__, args=[word])
    df = df[df["to"] != USER]
    if normalize:
        return df.groupby("to").occurrence.mean().sort_values()
    else:
        return df.groupby("to").occurrence.sum().sort_values()


def word_usage_per_day(word, match_case=False, user=None):
    text = "text_splitted" if match_case else "text_normalized_splitted"
    df = messages_df[messages_df["from"] == USER]
    if user:
        df = df[df["to"] == user]
    return df.groupby("day_session")[text].apply(lambda x: x.apply(list.__contains__, args=[word]).sum())


def word_first_occurrence(word, match_case=False, user=None):
    text = "text_splitted" if match_case else "text_normalized_splitted"
    df = messages_df[messages_df["from"] == USER]
    if user:
        df = df[df["to"] == user]
    df = df[df[text].apply(list.__contains__, args=[word])].sort_values("date")

    return df.iloc[0] if len(df) else None


def word_of_period(period="day", user=None):
    df = words_df[~words_df.stopword].copy()
    if user:
        df = df[df.chat_name == user]
    else:
        df = df[df["from"] == USER]
    period_cols = ["year"]
    df["year"] = df.day_session.apply(getattr, args=["year"])
    if period == "month":
        df["month"] = df.day_session.apply(getattr, args=["month"])
        period_cols.append("month")
    elif period == "week":
        df["week"] = df.day_session.apply(lambda x: datetime.date.isocalendar(x)[1])
        period_cols.append("week")
    elif period == "day":
        df["month"] = df.day_session.apply(getattr, args=["month"])
        df["day"] = df.day_session.apply(getattr, args=["day"])
        period_cols.extend(["month", "day"])
    return df.groupby(period_cols).word_normalized.apply(lambda x: x.value_counts().index[0])


def word_suggestions_by_user(user):
    df = words_df[(words_df["to"] == user) & words_df.relevant]
    tf = np.log2(df.groupby("word_normalized").apply(len))
    idf = np.log2(1 + words_unique_df[words_unique_df.word_normalized.isin(tf.index)].drop_duplicates(
        "word_normalized").sort_values("word_normalized").counts_normalized)
    idf.index = tf.index
    return (tf / idf).sort_values(ascending=False)


def word_distribution_by_users(word, sent=True, match_case=False):
    col_word = "word" + "_normalized" * match_case
    col_user = "to" if sent else "from"
    df = words_df[(words_df[col_word] == word) & (words_df[col_user] != USER)]
    word_usage = len(df)
    by_users = pd.DataFrame(df[col_user].value_counts())
    by_users.columns = ["Counts"]
    by_users["user"] = by_users.index
    by_users["Percentage in all usages"] = by_users.Counts / word_usage
    by_users["Percentage in distinct words"] = by_users.Counts / by_users.apply(lambda x: (words_df[col_user] == x.user).sum(), 1)
    return by_users.drop("user", axis=1)


def word_distribution_in_chat(word, chat_name, match_case=False):
    col = "text" + "_normalized" * match_case + "_splitted"
    df = messages_df[(messages_df.chat_name == chat_name)]
    df = df[df[col].apply(list.__contains__, args=[word])]
    return df.groupby(["day_session", "from"]).apply(len)


def longest_strike_by_count(user=None, max_break=15):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
    diffs = df.date.diff()
    diffs = diffs[diffs < datetime.timedelta(minutes=max_break)]
    lengths = [list(group) for group in mit.consecutive_groups(diffs.index)]
    longest = max(lengths, key=len)
    if lengths:
        return df.day_session[longest[0]], len(longest), df.date[longest[-1]] - df.date[longest[0]], longest
    else:
        return None, 0, datetime.timedelta(0), []



def longest_strike_by_duration(user=None, max_break=15):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
    diffs = df.date.diff()
    diffs = diffs[diffs < datetime.timedelta(minutes=max_break)]
    lengths = [list(group) for group in mit.consecutive_groups(diffs.index)]
    longest = max(lengths, key=lambda x: df.date[x[-1]] - df.date[x[0]])
    if lengths:
        return df.day_session[longest[0]], len(longest), df.date[longest[-1]] - df.date[longest[0]], longest
    else:
        return None, 0, datetime.timedelta(0), []


def longest_strike_by_days(users=None):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    if users:
        df = df[df.chat_name.isin(users)]
    return df.groupby("chat_name").apply(lambda x: longest_consecutive_days(x.day_session.unique())).apply(
        lambda x: pd.Series({"start": x[0], "end": x[1], "days": (x[1] - x[0]).days})).sort_values("days")[::-1]


def avg_message_length_per_day(users=None, sent=True):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    df = df[df["from" if sent else "to"] == USER]
    if users:
        df = df[df.chat_name.isin(users)]
    df = df.groupby(["day_session", "chat_name"]).text_splitted.apply(lambda x: x.apply(len).mean())
    df.columns = ["Word count"]
    return df


def message_count_per_day(users=None, sent=True):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    df = df[df["from" if sent else "to"] == USER]
    if users:
        df = df[df.chat_name.isin(users)]
    df = df.groupby(["day_session", "chat_name"]).apply(len)
    df.columns = ["Quantity"]
    return df


def message_percentage_by_unit(unit="hour", users=None, sent=True, grouped_by=None, plot=False):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    df = df[df["from" if sent else "to"] == USER]
    if users:
        df = df[df.chat_name.isin(users)]

    if unit == "month":
        df[unit] = df.date.dt.month.values
    elif unit == "weekday":
        df[unit] = df.date.dt.weekday.values
    elif unit == "hour":
        df[unit] = df.date.dt.hour.values
    else:
        raise Exception("Unit must be one of month, weekday, or hour.")

    if grouped_by:
        if grouped_by == "month":
            df[grouped_by] = df.date.dt.month.values
        elif grouped_by == "year":
            df[grouped_by] = df.date.dt.year.values
        if plot:
            fig = px.violin(df, x=grouped_by, y=unit)
        else:
            fig = None

    df = df[unit].value_counts(normalize=True).sort_index()
    if unit == "weekday":
        df.index = WEEKDAYS
    elif unit == "month":
        df.index = MONTHS

    return fig, df


def conversation_starters_and_enders_per_day(user):
    df = messages_df[(messages_df.type == "message") & (messages_df.chat_name == user)].sort_values("date")
    return df.groupby("day_session")["from"].apply(lambda x: x.iloc[0]), \
           df.groupby("day_session")["from"].apply(lambda x: x.iloc[-1])


def conversation_starts_and_ends(user=None, return_first=10):
    df = messages_df[(messages_df.type == "message")].query("text_normalized.str.strip() != ''").sort_values("date")
    if user:
        df = df[df.chat_name == user]
    return df.groupby("day_session").text_normalized.apply(lambda x: x.iloc[0]).value_counts().head(return_first), \
           df.groupby("day_session").text_normalized.apply(lambda x: x.iloc[-1]).value_counts().head(return_first)


def conversation_starting_and_ending_per_day(user):
    df = messages_df[(messages_df.type == "message") & (messages_df.chat_name == user)].sort_values("date")
    return df.groupby("day_session").date.apply(lambda x: x.iloc[0]), \
           df.groupby("day_session").date.apply(lambda x: x.iloc[-1])


def avg_reply_time_per_day(users=None, linkage="complete", questions_only=False, sent=True):
    col = "from" if sent else "to"
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    if users:
        df = df[df.chat_name.isin(users)]
    df.index = range(len(df))

    indices = [list(i) for i in mit.consecutive_groups(df[df[col] == USER].index)]

    if questions_only:
        indices = [group for group in indices if max(df.question_mark[group])]
    if linkage == "complete":
        indices = [[group[0], group[-1]] for group in indices]
    elif linkage == "single":
        indices = [[group[-1], group[-1]] for group in indices]
    else:
        raise Exception("Either a `complete` or a `single` linkage needs to be specified.")

    df["replies"] = None
    for i, j in indices:
        if (j + 1 < len(df)) and (df.day_session[i] == df.day_session[j + 1]):
            df.replies[j] = df.date[j + 1] - df.date[i]
    df = df.dropna(subset=["replies"]).groupby(["day_session", "chat_name"]).replies.mean()
    df.columns = ["Minutes"]

    return df


def max_parallel_chats(window=15):
    # returns (day_session, how many messages, how many people, people's names, ids of messages)
    # TODO: chats are not included
    df = messages_df[
        (messages_df.type == "message") & (messages_df["from"] == USER) & ~messages_df["to"].isna()].sort_values(
        "date")
    df["previous_to"] = df["to"].shift(1)
    df["next_to"] = df["to"].shift(-1)
    df = df[(df.previous_to != df["to"]) | (df["to"] != df.next_to)]
    df.index = range(len(df))
    df["indx"] = df.index

    df["max_stride"] = df.date_unixtime.apply(lambda x: (df.date_unixtime <= x + window * 60).sum()) - df.index
    df["n_parallel_chats"] = df.apply(lambda x: df.iloc[x.indx:x.indx + x.max_stride]["to"].nunique(), axis=1)
    max_parallel_chats = df.n_parallel_chats.max()
    parallel_chat_idx = [next(i) for i in mit.consecutive_groups(df.index[df.n_parallel_chats == max_parallel_chats])]

    parallel_chats = []
    for i in parallel_chat_idx:
        chats = df.iloc[i:i + df.max_stride[i]]
        parallel_chats.append((
            chats.day_session.iloc[0],
            len(chats),
            chats["to"].nunique(),
            chats["to"].unique(),
            chats.id.values
        ))

    return parallel_chats


def screen_time_per_day(offset=1, max_offset=5, user=None):
    # TODO: do something character-level
    df = messages_df[(messages_df.type == "message") & (messages_df["from"] == USER)].sort_values("date")
    if user:
        df = df[df.chat_name == user]
    df.index = range(len(df))
    diffs = -df.date.diff(-1)
    df["end_date"] = df.date + datetime.timedelta(minutes=offset)
    df.end_date[diffs < datetime.timedelta(minutes=max_offset)] = df.date.shift(-1)
    return df.groupby("day_session")[["date", "end_date"]].apply(lambda x: x.diff(axis=1).end_date.sum())


def emotion_per_day(user=None):
    df = messages_df[messages_df.type == "message"]
    if user:
        df = df[df.chat_name == user]
    emotion_labels = [f"emotion_{i}" for i in EMOTIONS]

    return df.groupby(["day_session", "from"])[emotion_labels].mean(), \
           df.groupby(["day_session", "from"])[emotion_labels].sum()


def longest_gap(users=None):
    df = messages_df[(messages_df.type == "message") & messages_df.pm]
    df.index = df.id
    if users:
        df = df[df.chat_name.isin(users)]
    gaps = df.groupby("chat_name").day_session.apply(lambda x: x.append(pd.Series([TODAY])).diff().max())
    gaps_idx = df.groupby("chat_name").day_session.apply(lambda x: x.append(pd.Series([TODAY])).diff().argmax())
    last_messages = []
    for i in gaps.index:
        msg = df[df.chat_name == i].iloc[gaps_idx[i] - 1]
        last_messages.append({
            "chat_name": i,
            "gap": gaps[i],
            "date": msg.date,
            "day_session": msg.day_session,
            "text": msg.text,
            "from": msg["from"],
            "to": msg["to"],
            "id": msg.id
        })
    return pd.DataFrame(last_messages)


def disappeared_by_day(max_gap=50):
    cols = ["gap", "day_session", "text", "from"]
    df = messages_df[(messages_df.type == "message") & messages_df.pm & (messages_df["from"] != USER)]
    df = df.groupby("from").apply(lambda x: x.iloc[-1])
    df["gap"] = (TODAY - df.day_session).apply(lambda x: x.days)
    df = df[df.gap > max_gap][cols].sort_values("gap")
    df.columns = ["Days since", "Last communicated", "Last message", "Sender"]
    return df


def stickers_per_day(user=None, sent=True):
    df = messages_df[(messages_df.type == "sticker") & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]

    df = df.groupby("day_session")[[col, "sticker_emoji"]].apply(lambda x: pd.Series(
                                                                            [len(x), mod(x.sticker_emoji.dropna())]))
    df.columns = ["counts", "mod_emoji"]
    return df


def images_per_day(user=None, sent=True):
    df = messages_df[messages_df.photo & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]
    df = df.groupby("day_session")[col].apply(len)
    df.columns = ["counts"]
    return df


def gifs_per_day(user=None, sent=True):
    df = messages_df[messages_df.gif & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]
    df = df.groupby("day_session")[col].apply(len)
    df.columns = ["counts"]
    return df


def voices_per_day(user=None, sent=True):
    df = messages_df[messages_df.voice & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]
    df = df.groupby("day_session")[col].apply(len)
    df.columns = ["counts"]
    return df


def videos_per_day(user=None, sent=True):
    df = messages_df[messages_df.video & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]
    df = df.groupby("day_session")[col].apply(len)
    df.columns = ["counts"]
    return df


def files_per_day(user=None, sent=True):
    df = messages_df[messages_df.file & messages_df.pm]
    if user:
        df = df[df.chat_name == user]
        col = "from"
    if sent:
        col = "to"
        df = df[df["from"] == USER]
    else:
        col = "from"
        df = df[df["to"] == USER]
    df = df.groupby("day_session")[col].apply(len)
    df.columns = ["counts"]
    return df


def message_groups_per_day(user=None):
    # ill-written function
    df = messages_df[
        (messages_df.type == "message") & (messages_df["from"] == USER) & ~messages_df["to"].isna()].sort_values(
        "date")
    if user:
        df = df[df["to"] == user]
        return df.groupby("day_session").id.apply(lambda x: np.mean([len(list(i)) for i in mit.consecutive_groups(x)]))
    else:
        df.groupby("day_session").apply(lambda x: x.groupby("to").id.apply(
            lambda x: [len(list(i)) for i in mit.consecutive_groups(x)]).sum()).apply(np.mean)


def special_names_per_day(user=None, exclude=[]):
    # all mentions of specials
    df = messages_df[messages_df.special_names.apply(len).astype(bool)]
    if user:
        df = df[df.chat_name == user]
        exclude.append(user)
    df.special_names = df.special_names.apply(lambda x: [i for i in x if i not in exclude])
    df = df[df.special_names.apply(len).astype(bool)]
    df = df.groupby("day_session").special_names.apply(lambda x: mod(x.sum(), return_counts=True)).dropna().apply(pd.Series)
    df.columns = ["name", "counts"]
    return df


def most_forwarded_from_per_day(user=None):
    df = messages_df[
        (messages_df.type == "message") & (messages_df["from"] == USER) & ~messages_df.forwarded_from.isna()]
    if user:
        df = df[df["to"] == user]
    df = df.groupby("day_session").forwarded_from.apply(mod, return_counts=True).apply(pd.Series)
    df.columns = ["name", "counts"]
    return df


def most_forwarded_to_per_day():
    df = messages_df[(messages_df.type == "message") & (
            messages_df["from"] == USER) & messages_df.pm & ~messages_df.forwarded_from.isna()]
    df = df.groupby("day_session")["to"].apply(mod, return_counts=True).apply(pd.Series)
    df.columns = ["name", "counts"]
    return df


def chat_adders():
    return messages_df[~messages_df.pm].groupby("chat_name").actor.apply(lambda x: x.iloc[0]).value_counts().drop(USER)


def interlocutors_per_day(sent=True):
    df = messages_df[messages_df.pm]
    df = df[df["from" if sent else "to"] == USER]
    return df.groupby("day_session").chat_name.unique()


def notable_days(user=None, max_break=15, offset=1, max_offset=5, window=15):
    df = messages_df[messages_df.type == "message"]
    if user:
        df = df[df.chat_name == user]

    days = {}
    days["longest_conversation_by_count"] = longest_strike_by_count(user=user, max_break=max_break)[0]
    days["longest_conversation_by_duration"] = longest_strike_by_duration(user=user, max_break=max_break)[0]
    days["longest_screen_time"] = \
        screen_time_per_day(user=user, offset=offset, max_offset=max_offset).sort_values().index[-1]
    days["most_messages_sent"] = flatten_multiindex(message_count_per_day(users=[user] if user else None)).sort_values().index[-1]
    days["most_messages_received"] = \
        flatten_multiindex(message_count_per_day(users=[user] if user else None, sent=False)).sort_values().index[-1]
    days["most_conversations_at_once"] = max_parallel_chats(window=window)[0][0]
    days["first_message_sent"] = df.day_session[df["from"] == USER].min()
    days["first_message_received"] = df.day_session[df["to"] == USER].min()

    return days


def emotion_per_chat(users=None, min_messages=400):
    df = messages_df[messages_df.pm]
    if users:
        df = df[df.chat_name.isin(users)]
    chats = messages_df.chat_name.value_counts()
    chats = chats.index[chats >= min_messages]
    df = df[df.chat_name.isin(chats)]
    df = df.groupby("chat_name")[['emotion_happy', 'emotion_sad', 'emotion_funny', 'emotion_angry',
                                    'emotion_lovely', 'emotion_surprised']].mean()
    df.columns = [i.split("_")[1] for i in df.columns]
    return df


def emotion_per_day(user=None):
    df = messages_df[messages_df.type == "message"]
    if user:
        df = df[df["chat_name"] == user]
    else:
        df["from"][df["from"] == USER] = "others"
    emotion_labels = [f"emotion_{i}" for i in EMOTIONS]

    return df.groupby(["day_session", "from"])[emotion_labels].mean(), \
           df.groupby(["day_session", "from"])[emotion_labels].sum()


def command_handler(command_id):
    if command_id == 1:
        option_unit = st.radio("Sort by:", ["hours", "weekdays", "months"], horizontal=True)
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_groupby = st.radio("Group clusters by:", ["months", "years"], horizontal=True)

        col1, col2 = st.columns(2)
        fig, result = message_percentage_by_unit(unit=option_unit[:-1], users=option_users, sent=option_sent == "by me",
                                                 plot=True, grouped_by=option_groupby[:-1])
        col1.plotly_chart(fig, use_container_width=True)
        col2.dataframe(result)

    elif command_id == 2:
        option_user = st.selectbox("Select the chat:", USERS)
        st.dataframe(word_suggestions_by_user(user=option_user))

    elif command_id == 3:
        st.markdown(
            "`Hint:` You can input either one or more words per field. In case of multiple words, separate them with commas (e.g. `hey, hello` vs `goodbye`).")
        option_a_s = st.text_input('Compare these words:').replace(", ", ",").split(",")
        option_b_s = st.text_input('with these:').replace(", ", ",").split(",")
        option_match_case = st.checkbox('Match case')
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        a, b = a_vs_b(option_a_s, option_b_s, word_usage_per_day, agg_func="sum", match_case=option_match_case,
                      user=option_user)
        df = pd.DataFrame([a, b]).fillna(0).T
        df.columns = ["Set 1", "Set 2"]
        st.dataframe(hide_zeros(df).astype(int))
        df["date"] = df.index
        fig = px.line(df, x="date", y=["Set 1", "Set 2"])
        st.plotly_chart(fig)

    elif command_id == 4:
        option_users = st.multiselect("Limit to the following chats (optional):", ["(all)"] + USERS)
        longest = longest_gap(option_users).iloc[0]
        st.markdown(f"""The longest gap ever began from `{longest.day_session}` and continued for `{longest.gap.days}` days.
During this period, there was no message exchanged between you and `{longest.chat_name}`, the last one being "{longest.text}" \
sent by {longest["from"]} to {longest["to"]}.""")

    elif command_id == 5:
        option_word = st.text_input("Word:")
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        option_match_case = st.checkbox('Match case')
        df = word_distribution_by_users(option_word, option_sent, option_match_case)
        st.dataframe(df)

    elif command_id == 6:
        option_word = st.text_input("Word:")
        option_match_case = st.checkbox('Match case')
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        option_groupby = st.radio("Show usage per:", ["day", "week", "month", "year"])
        df = word_usage_per_day(option_word, option_match_case, option_user)
        if option_groupby != "day":
            df = groupby(df, option_groupby)
        df.columns = ["Quantity"]
        st.dataframe(df)

    elif command_id == 7:
        option_user = st.selectbox("Select the chat:", USERS)
        df = pd.DataFrame(conversation_starting_and_ending_per_day(option_user)).T
        df.columns = ["Start", "End"]
        # fig = ff.create_distplot([df.Start.values, df.End.values], df.columns)
        df.Start = df.Start.dt.strftime("%r")
        df.End = df.End.dt.strftime("%r")
        st.dataframe(df)
        # st.plotly_chart(fig, use_container_width=True)

    elif command_id == 8:
        option_user = st.selectbox("Select the chat:", USERS)
        df = pd.DataFrame(conversation_starters_and_enders_per_day(option_user)).T
        df.columns = ["Starter", "Ender"]
        st.dataframe(df)
        col1, col2 = st.columns(2)
        col1.dataframe(df.Starter.value_counts())
        col2.dataframe(df.Ender.value_counts())
        # st.write("Or, if we group by the month...")
        # st.dataframe(groupby(df.starter, "month", "sum", True))
        # fig1 = go.Figure(data=[
        #     go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]),
        #     go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])
        # ])

    elif command_id == 9:
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        option_return_first = st.number_input("Show most popular:", value=10)
        df = conversation_starts_and_ends(option_user, option_return_first)
        col1, col2 = st.columns(2)
        col1.write("Starting")
        col1.dataframe(df[0])
        col2.write("Ending")
        col2.dataframe(df[1])

    elif command_id == 10:
        option_offset = st.number_input("Min screen time of a message (in minutes)", value=1)
        option_max_offset = st.number_input("Max screen time of a message (in minutes)", value=5)
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        df = screen_time_per_day(option_offset, option_max_offset, option_user).dt.total_seconds() // 60
        df.columns = ["Minutes"]
        st.dataframe(df.astype(int))
        df["day"] = df.index
        fig = px.line(df, x="day", y="Minutes")
        st.plotly_chart(fig)

    elif command_id == 11:
        option_window = st.number_input("During a window of (in minutes)", value=15)
        day, count_m, count_p, people, messages = max_parallel_chats(option_window)[0]
        st.markdown(f"You held a maximum number of **{count_p}** conversations at once with `{'`, `'.join(people)}` on **{day}**, exchanging **{count_m}** messages.")

    elif command_id == 12:
        option_media_type = st.radio("Media type:", ["sticker", "image", "gif", "voice", "video", "file"])
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        func = eval(option_media_type + "s_per_day")
        df = func(option_user, option_sent)
        df.columns = ["Quantity"]
        st.dataframe(df)
        df["day"] = df.index
        fig = px.line(df, x="day", y="Quantity")
        st.plotly_chart(fig)

    elif command_id == 13:
        df = chat_adders()
        df.columns = ["Chats"]
        if len(df):
            st.markdown(f"As you always knew, it's `{df.index[0]}` who added to you to those {df.iloc[0]} group chats.")
            st.dataframe(df)
        else:
            st.markdown("~~Un~~fortunately, no one added you to a chat.")

    elif command_id == 14:
        option_max_gap = st.number_input("Minimum number of days since disappearance:", value=50)
        df = disappeared_by_day(option_max_gap)
        st.dataframe(df)

    elif command_id == 15:
        st.markdown(
            "`Hint:` In case of multiple special names to be ignored, separate them with commas. Case doesn't matter.")
        option_exclude = st.text_input('Special names to ignore:').replace(", ", ",").split(",")
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        df = special_names_per_day(option_user, option_exclude)
        st.dataframe(df)
        st.write("")
        st.write("Overall speaking:")
        st.dataframe(df.groupby("name").counts.apply(sum).sort_values()[::-1])

    elif command_id == 16:
        st.markdown(
            "`Hint:` To configure the output, change the values below. To get more detailed information about any of \
            the days, refer to the specific tool from the dropdown below.")
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        option_max_break = st.number_input("Max break in a continuous strike (in minutes)", value=15)
        option_offset = st.number_input("Min screen time of a message (in minutes)", value=1)
        option_max_offset = st.number_input("Max screen time of a message (in minutes)", value=5)
        option_window = st.number_input("Conversations at once in a window of (in minutes)", value=15)
        days = notable_days(option_user, option_max_break, option_offset, option_max_offset, option_window)

        st.markdown(f"""- Your longest non-stop conversation (by count) happened on `{days["longest_conversation_by_count"]}`
- While on `{days["longest_conversation_by_duration"]}` you did the lengthiest talking
- On `{days["most_conversations_at_once"]}` you did a combo by handling the most conversations at once
- And your longest active time on Telegram was registered right on `{days["longest_screen_time"]}`
- Since `{days["first_message_sent"]}` when you sent your first message
- You've set a record on sent messages on `{days["most_messages_sent"]}`
- Received an unprecedented number of them on `{days["most_messages_received"]}`
- With somebody texting you your very first message on `{days["first_message_sent"]}`""")

    elif command_id == 17:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        df = message_count_per_day(option_users, option_sent)
        st.dataframe(df)

        option_smoothing_window = st.slider("Smoothing window:", 0, 20, 1)
        smooth = pd.DataFrame({i: complement_index(df.unstack()[i]) for i in df.unstack()}).fillna(0).astype(int)
        smooth = pd.DataFrame({i: smoothen(smooth[i], option_smoothing_window) for i in smooth}).fillna(0)
        fig = px.line(smooth)
        st.plotly_chart(fig)

    elif command_id == 18:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        df = avg_message_length_per_day(option_users, option_sent)
        st.dataframe(df)
        fig = px.line(df.unstack(), color="chat_name")
        st.plotly_chart(fig)

    elif command_id == 19:
        option_period = st.radio("Word of the:", ["day", "week", "month", "year"], horizontal=True)
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None
        df = word_of_period(option_period, option_user)
        st.dataframe(df)

    elif command_id == 20:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_linkage = st.radio("Reply time after the:", ["single", "complete"], format_func=lambda x: "last message in a row"
                                  if x == "complete" else "first message in a row")
        option_questions_only = st.checkbox("Only messages containing question marks")
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"

        df = avg_reply_time_per_day(option_users, option_linkage, option_questions_only, option_sent)
        st.dataframe(df)
        fig = px.line(df.unstack(), color="chat_name")
        st.plotly_chart(fig)

    elif command_id == 21:
        option_user = st.selectbox("Limit to the following chat (optional):", ["(all)"] + USERS)
        if option_user == "(all)":
            option_user = None

        df = most_forwarded_from_per_day(option_user)
        st.dataframe(df)
        st.dataframe(df.fillna(0).groupby("name").counts.sum())

    elif command_id == 22:
        df = most_forwarded_to_per_day()
        st.dataframe(df)
        st.dataframe(df.fillna(0).groupby("name").counts.sum())

    elif command_id == 23:
        option_word = st.text_input("Word to get stats for:").lower()
        if option_word in words_unique_df.index:
            st.markdown(f"""`{option_word}` was born on {words_unique_df.birth[option_word]} when it first time ever \
    occurred in a chat, and it was last used on {words_unique_df.death[option_word]} so far. Since then, it was used \
    {words_unique_df.counts_normalized[option_word]} times, with its peak day on {words_unique_df.peak_day[option_word]}.""")
            st.markdown(
                """`Hint:` You might also want to check "Word popularity by chats" and "Word usage per day/week/month/year\"""")
        else:
            if option_word:
                st.write("Oops! Unknown word...")

    elif command_id == 24:
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        df = questions_asked_per_user(option_sent)
        st.dataframe(df)

    elif command_id == 25:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_min_messages = st.number_input("Min messages in a chat to analyze:", 400, step=50)
        df = emotion_per_chat(option_users, option_min_messages)
        st.write("Here is the percentage of messages expressing each emotion, based on emojis and emoticons.")
        st.dataframe(df)

    elif command_id == 26:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        df = hide_zeros(longest_strike_by_days(option_users))
        st.dataframe(df)

    elif command_id == 27:
        option_users = st.multiselect("Limit to the following chats (optional):", USERS)
        option_sent = st.radio("For messages sent:", ["by me", "to me"], horizontal=True) == "by me"
        df = message_count_per_day(option_users, option_sent).unstack().fillna(0).astype(int)
        st.dataframe(df)
        df = pd.DataFrame({i: complement_index(df[i]).expanding().sum() for i in df}).astype(int)

        with st.spinner(text="We're working on a bar chart race..."):
            bcr_html = bcr.bar_chart_race(
                df=df,
                n_bars=5,
                bar_size=.95,
                figsize=(5, 3),
                #     dpi=144,
                title='Number of Messages with Top 5 People',
                bar_kwargs={'alpha': .7},
                filter_column_colors=True,
                steps_per_period=100,
                period_length=50,
                writer="html"
            )
            components.html(bcr_html.data)


file = st.file_uploader("Upload result.json file here:", type=['json'])


@st.cache(suppress_st_warning=True)
def preprocess(chat_length_threshold, canonical_form_threshold, special_name_threshold):
    global USER, USERS, chats_df, messages_df, words_df, words_unique_df, special_names_df
    with st.spinner(text='Chats are parsed...'):
        USER, chats_df = get_chats_df(file)

    with st.spinner(text='Messages are read...'):
        messages_df = get_messages_df(chat_length_threshold)

    with st.spinner(text='Words are extracted...'):
        words_df, words_unique_df, special_names_df = get_word_dfs(canonical_form_threshold, special_name_threshold)
        messages_df["special_names"] = messages_df.text_splitted.apply(
            lambda x: [special_names_df.canonical[i] for i in x if i in special_names_df.index])
        USERS = list(messages_df.query("pm").chat_name.unique())

        st.subheader("Great! Now your data is processed successfully.")
    return USER, USERS, chats_df, messages_df, words_df, words_unique_df, special_names_df


if file is not None:
    st.sidebar.write("Advanced settings:")
    chat_length_threshold = st.sidebar.number_input('Minimum number of messages in a chat:',
                                                    value=3, key="chat_length_threshold")
    canonical_form_threshold = st.sidebar.number_input('Minimum number of a form to be considered canonical:',
                                                       value=10, key="canonical_form_threshold")
    special_name_threshold = st.sidebar.number_input(
        'Minimum percentage of capital-case occurrences of a special name:',
        value=0.95, key="special_name_threshold")
    USER, USERS, chats_df, messages_df, words_df, words_unique_df, special_names_df = preprocess(chat_length_threshold,
                                                     canonical_form_threshold, special_name_threshold)

    with st.spinner(text='In progress'):
        command = st.selectbox("Please select what insights you'd like to be drawn from your data.",
                               format_func=OPTIONS.__getitem__, options=range(len(OPTIONS)))
        command_handler(command)
