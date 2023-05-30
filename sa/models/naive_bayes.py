from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV


def naive_bayes_mdl(df, log_level=2):

    # split data into train data 80% and test data 20%
    x_train, x_test, y_train, y_test = train_test_split(df['msg'], df['class'], test_size=0.2, random_state=30)

    pl = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())])

    # Set hyperparameter search space
    tuned_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        #This helps to avoid situations where certain words or features have zero probability
        'clf__alpha': [1e-3, 1e-2, 1e-1, 1]
    }

    # Conduct 5-fold cross-validation
    grid = GridSearchCV(pl, tuned_parameters, cv=5, verbose=log_level, n_jobs=-1)
    grid.fit(x_train, y_train)
    print(grid.best_estimator_)

    # Predict labels for test set using best model
    y_pred = grid.predict(x_test)

    return y_test, x_test, y_pred, grid.best_estimator_
