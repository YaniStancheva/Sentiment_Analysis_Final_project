from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def random_forest_mdl(df, log_level=2):

    # split data into train data 80% and test data 20%
    x_train, x_test, y_train, y_test = train_test_split(df['msg'], df["class"], test_size=0.2, random_state=30)
    # x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=30)

    pl = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', RandomForestClassifier())])

    # Set hyperparameter search space
    tuned_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        #the number of decision trees to include in the random forest classifier
        'clf__n_estimators': [10, 20, 50],
        #the minimum number of samples required to split an internal decision tree node during training.
        'clf__min_samples_split': [2, 5, 10],
        #only one weight is assigned
        'clf__class_weight': ['balanced'],
        #the maximum depth of each decision tree in the random forest classifier
        'clf__max_depth': [50],
        #whether or not to use bootstrapping (sampling with replacement) during training
        'clf__bootstrap': [False]
    }

    # Conduct 3-fold cross-validation
    grid = GridSearchCV(pl, tuned_parameters, cv=3, verbose=log_level, n_jobs=-1)
    grid.fit(x_train, y_train)
    print(grid.best_estimator_)

    # Predict labels for test set using best model
    y_pred = grid.predict(x_test)

    return y_test, x_test, y_pred, grid.best_estimator_

