
import re

from sa.preprocessing import CHATSLANGS, STOPWORDS

import nltk.downloader
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')


# %% Functions
def replace_list_item_by_dict_key(lst, dct, default_value=None):
    return list(map(lambda x: dct.get(x, default_value if default_value is not None else x), lst))


def replace_non_ascii_words(string):

    # Replaces words containing non-ASCII characters with an empty string.
    words = string.split()
    for i, word in enumerate(words):
        if any(ord(char) > 127 for char in word):
            words[i] = ''
    return ' '.join(words)


def message_prepare(df):

    # Removing white spaces and converting to lowercase
    df['message'] = df['message'].str.strip().str.lower()
    df['is_retweet'] = df['message'].str.startswith('rt')

    print('Replacing URLs, mentions, and hashtags with with URL')

    # replacing anything which starts with http, hashtag and @ sign with URL
    df['msg_rep'] = df['message'].astype(str).apply(lambda x: re.sub(r'http\S+|#\S+|@\S+', 'URL', x))

    # removing messages with
    # df['msg_rep'].replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
    df['msg_rep'] = df['msg_rep'].apply(replace_non_ascii_words)

    # replacing retweet rt tags from messages
    df['msg_rep'] = df['msg_rep'].apply(lambda x: re.sub(r'rt URL', 'URL', x))

    # removing all URLs from msg
    df['msg'] = df['msg_rep'].str.replace(r'URL', '', regex=True)
    df['msg'] = df['msg_rep'].str.replace(r'\n', ' ', regex=True)

    # Removing duplicate messages and printing the outputs
    print('Removing duplicate messages...')
    print(len(df))
    print(df['msg'].duplicated().value_counts())
    print(df['msg'].duplicated().value_counts(normalize=True))
    df.drop_duplicates(subset=['msg'], inplace=True)

    return df


def tokenize(df):
    tokenizer = RegexpTokenizer(r'\w+')
    df['words'] = df['msg'].apply(lambda x: tokenizer.tokenize(x))
    return df


def remove_stopwords(df):
    additional_stopwords = ['URL']
    stop_words = STOPWORDS + additional_stopwords
    df['words'] = df['words'].apply(lambda x: list(filter(lambda xx: xx not in stop_words, x)))
    return df


def translate_slang(df):
    df['words'] = df['words'].apply(replace_list_item_by_dict_key, dct=CHATSLANGS)
    return df


def lemmatize(df):
    wnl = WordNetLemmatizer()
    df['words'] = df['words'].apply(lambda x: list(map(wnl.lemmatize, x)))
    return df


def remove_empty(df):
    df = df.loc[~df['words'].apply(lambda x: not x)]
    df = df.loc[df['class'].notna()]
    return df


def re_join(df):
    df['msg'] = df['words'].apply(lambda x: ' '.join(x))
    return df

def run_preprocessing_pipeline(df):
    return df.pipe(message_prepare)\
        .pipe(tokenize)\
        .pipe(remove_stopwords)\
        .pipe(translate_slang)\
        .pipe(lemmatize)\
        .pipe(remove_empty)\
        .pipe(re_join)
