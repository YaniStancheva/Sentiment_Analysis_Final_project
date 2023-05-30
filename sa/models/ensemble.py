from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier


def ensemble_mdl(df, estimators):

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(df['msg'], df['class'], test_size=0.2, random_state=42)

    # Define the stacking classifier
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return y_test, y_pred