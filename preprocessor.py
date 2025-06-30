import re
import pandas as pd
import emoji
from urlextract import URLExtract

def preprocess(data):
    # Pattern for date-time in 12-hour format
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[apAP][mM]\s-\s)'
    parts = re.split(pattern, data)
    messages = [part for part in parts if part and not re.match(pattern, part)]
    dates = [date.replace('\u202f', ' ') for date in re.findall(pattern, data)]

    df = pd.DataFrame({'user_message': messages, 'date': dates})
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y, %I:%M %p - ', errors='coerce')

    # Extract user and message
    pattern_for_username = r'([^:]+):'
    df['user'] = df['user_message'].str.extract(pattern_for_username)
    df['message'] = df['user_message'].str.replace(pattern_for_username, '', regex=True).str.strip()
    df.drop(columns=['user_message'], inplace=True)

    # Drop system messages or messages without a user
    df = df.dropna(subset=['user'])

    # Feature engineering
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['time'] = df['date'].dt.strftime('%I:%M %p')

    df['total_word'] = df['message'].apply(lambda x: len(x.split()))
    df['url_count'] = df['message'].apply(lambda x: len(URLExtract().find_urls(x)))
    df['emoji_count'] = df['message'].apply(emoji.emoji_count)

    # Period classification for heatmap
    df['period'] = df['hour'].apply(
        lambda x: 'Night' if 0 <= x < 6 else (
            'Morning' if 6 <= x < 12 else (
                'Afternoon' if 12 <= x < 18 else 'Evening'
            )
        )
    )

    return df
