from wordcloud import WordCloud
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import emoji
from collections import Counter
from urlextract import URLExtract


def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    total_messages = df.shape[0]
    total_word_count = df['total_word'].sum()
    total_media_messages = df[df['message'] == '<Media omitted>'].shape[0]
    total_url_count = df['url_count'].sum()
    total_emoji_count = df['emoji_count'].sum()

    deleted_message = df[df['message'].str.contains("This message was deleted", na=False)]['message']
    edited_messages = df[df['message'].str.contains("<This message was edited>", na=False)]['message']

    phone_pattern = r'\+?\d{2,4}[\s-]?\d{10}'
    shared_contacts = df[df['message'].str.contains(phone_pattern, case=False, na=False) |
                         df['message'].str.contains('.vcf', case=False, na=False)]['message']

    location_pattern = r'//maps\.google\.com/\?q=\d+\.\d+,\d+\.\d+'
    shared_locations = df[df['message'].str.contains(location_pattern, case=False, na=False)]['message']

    return (
        total_messages,
        total_word_count,
        total_media_messages,
        total_url_count,
        total_emoji_count,
        len(deleted_message),
        len(edited_messages),
        len(shared_contacts),
        len(shared_locations)
    )


def extract_sentiment(message):
    analysis = TextBlob(message)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


def perform_tfidf_analysis(messages):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = vectorizer.fit_transform(messages)
    words = vectorizer.get_feature_names_out()
    top_n = 5
    row_id = np.argmax(tfidf.toarray(), axis=0)
    top_words = [(words[i], tfidf[row_id[i], i]) for i in np.argsort(-tfidf.toarray().sum(axis=0))[:top_n]]
    return top_words


def perform_lda_analysis(messages, num_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    bow = vectorizer.fit_transform(messages)
    words = vectorizer.get_feature_names_out()
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(bow)
    topic_words = []
    for i, topic in enumerate(lda.components_):
        top_words_array = topic.argsort()[-5:][::-1]
        topic_list = [words[j] for j in top_words_array]
        topic_words.append(f"Topic {i + 1}: {' | '.join(topic_list)}")
    return topic_words


def perform_comparative_analysis(df, users_to_compare, start_date, end_date):
    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date + pd.Timedelta(days=1))
    date_filtered_df = df[(df["date"].dt.date >= start_date) & (df["date"].dt.date < end_date)]
    user_filtered_df = date_filtered_df[date_filtered_df["user"].isin(users_to_compare)]
    return user_filtered_df["user"].value_counts()


def most_least_busy_users(df):
    message_counts = df['user'].value_counts()
    top_users = message_counts.head(5)
    bottom_users = message_counts.tail(5)
    return top_users, bottom_users


def user_activity_over_time(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby(['date', 'user'])['message'].count().unstack().fillna(0)


def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return df['day'].value_counts().reindex(ordered_days)


def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.pivot_table(index='day', columns='period', values='message', aggfunc='count').fillna(0)


# def create_wordcloud(selected_user, df, stopwords_path='stop_hinglish.txt'):
#     with open(stopwords_path, 'r') as f:
#         stop_words = set(f.read().split())

#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]

#     df = df[~df['message'].isin(['<Media omitted>', 'This message was deleted', '<This message was edited>'])]
#     wc = WordCloud(width=800, height=400, min_font_size=10, background_color='white', stopwords=stop_words)
#     df_wc = wc.generate(' '.join(df['message']))
#     return df_wc.to_image()


# def emoji_helper(selected_user, df):
#     if selected_user != 'Overall':
#         df = df[df['user'] == selected_user]

#     all_possible_emojis = set(emoji.emojize(alias) for alias in emoji.unicode_codes.EMOJI_DATA.keys())

#     all_emojis = []
#     for message in df['message']:
#         message_str = str(message)
#         all_emojis.extend([char for char in message_str if char in all_possible_emojis])

#     return pd.DataFrame(Counter(all_emojis).most_common(), columns=['Emoji', 'Frequency'])


def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date').count()['message'].reset_index()
