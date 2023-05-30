from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def linear_svc_mdl(df, log_level=2):

    # split data into train data 80% and test data 20%
    x_train, x_test, y_train, y_test = train_test_split(df['msg'], df['class'], test_size=0.2, random_state=30)
                            #converts data into tokens(numerical features that can be understood by the machine learning models)
    pl = Pipeline([('vect', CountVectorizer()),
                            #converts raw text documents into feature vectors
                   ('tfidf', TfidfTransformer()),
                   ('clf', LinearSVC())])

    # Set hyperparameter search space
    tuned_parameters = {
        #specifies the n-gram range from the text
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2), (1, 3)],
        #specifies whether to use inverse document frequency
        'tfidf__use_idf': (True, False),
        #types of normalization 
        'tfidf__norm': ('l1', 'l2'),
        #regularization parameter. Smaller value means stronger regularization, larger ones means weaker
        'clf__C': [0.05, 0.1, 0.5, 1],
        #spefcifies the loss function
        'clf__loss': ['squared_hinge'],
        #class weight during training. 'balanced' indicates that the weights will be calculated based on the class frequencies in the training data
        'clf__class_weight': ['balanced']
    }

    # Conduct 5-fold cross-validation
    #pl-> estimator object
    #cv -> number of folds to be used during  cross-validation
    #verbose -> controls how detailed is the output
    #n_jobs ->the number of CPUs to be used, -1 means all of them
    grid = GridSearchCV(pl, tuned_parameters, cv=5, verbose=log_level, n_jobs=-1)
    grid.fit(x_train, y_train)
    print(grid.best_estimator_)

    # Predict labels for test set using best model
    y_pred = grid.predict(x_test)

    return y_test, x_test, y_pred, grid.best_estimator_
