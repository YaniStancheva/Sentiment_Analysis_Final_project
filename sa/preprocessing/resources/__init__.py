import pandas as pd
from nltk.corpus import stopwords

CHATSLANGS = pd.read_csv('./sa/preprocessing/resources/chat_slangs.csv')
STOPWORDS = stopwords.words('english')
