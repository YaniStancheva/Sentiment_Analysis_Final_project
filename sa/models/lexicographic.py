import pandas as pd
import numpy as np


# %% Functions
def replace_list_item_by_dict_key(lst, dct, default_value=None):
    #For each element in lst, the function looks up the corresponding value in dct and returns that value if it exists, or default_value if it does not exist in the dictionary
    return list(map(lambda x: dct.get(x, default_value if default_value is not None else x), lst))


def get_sentiment_score(df, lex):
    df['snt_score'] = df['words'].apply(lambda x: sum(replace_list_item_by_dict_key(lst=x, dct=lex, default_value=0)))
    #values are binned into three categories (-1, 0, 1)
    df['snt_pred'] = pd.cut(df['snt_score'], bins=[-np.inf, 0, 1, np.inf], labels=[-1, 0, 1]).astype(int)
    return df


def lexicographic_mdl(df):
    #the second column of the file is extracted as a dictionary
    lex = pd.read_csv('./sa/models/AFINN-111.txt', sep='\t', header=None, index_col=0)[1].to_dict()
    df = df.pipe(get_sentiment_score, lex=lex)
    return df[['snt_score', 'snt_pred']]
