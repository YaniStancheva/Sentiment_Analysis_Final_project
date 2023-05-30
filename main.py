import pandas as pd

from sa.models import lexicographic_mdl, naive_bayes_mdl, linear_svc_mdl, random_forest_mdl
from sa.models.ensemble import ensemble_mdl
from sa.preprocessing.preprocessing import run_preprocessing_pipeline
from sklearn.metrics import classification_report
import joblib

# If model training should be done True/False
RUN_MODELS = True

# If ensemble model training should be done True/False
RUN_ENSEMBLE = True

# Which models should be run nb/lsvc/rf/lx
MODELS = ['lx', 'nb', 'lsvc', 'rf']

# Which datasets should the models train on 'si'/'cc'
DS = ['cc', 'si']

# Output level to STDIO
LOG_LEVEL = True

if __name__ == '__main__':

    if 'cc' in DS:

        # %% Data import stage
        fp = './sa/sample-data/twitter_sentiment_data.csv'
        df = pd.read_csv(fp).rename(columns={'sentiment': 'class'})
        df = df.loc[df['class'] != 2]

        # %% Preprocessing
        df = run_preprocessing_pipeline(df)

    if RUN_MODELS and 'cc' in DS:

        if 'lx' in MODELS:
            # Lexicographic
            print('Running the lexographic model...')
            lx = lexicographic_mdl(df)
            print(classification_report(df['class'], lx['snt_pred']))

        if 'nb' in MODELS:
            # Naive Bayes
            print('Running the Naive Bayes model for climate change dataset...')
            nb_test, nb_x_test, nb_pred, nb_model_cc = naive_bayes_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(nb_test, nb_pred))
            joblib.dump(nb_model_cc, 'nb_model_cc.pkl')

        if 'lsvc' in MODELS:
            # Linear svc
            print('Running the Linear SVC model for climate change dataset...')
            lsvc_test, lsvc_x_test, lsvc_pred, lsvc_model_cc = linear_svc_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(lsvc_test, lsvc_pred))
            joblib.dump(lsvc_model_cc, 'lsvc_model_cc.pkl')

        if 'rf' in MODELS:
            # Random forest
            print('Running the Random Forest model...')
            rf_test, rf_x_test, rf_pred, rf_model_cc = random_forest_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(rf_test, rf_pred))
            joblib.dump(nb_model_cc, 'rf_model_cc.pkl')

    if RUN_ENSEMBLE and 'cc' in DS:

        print('Running Ensemble Classifier for climate change dataset...')
        models = [('nb', joblib.load('nb_model_cc.pkl')), ('lsvc', joblib.load('lsvc_model_cc.pkl'))]
        ens_test, ens_pred = ensemble_mdl(df, estimators=models)
        print(classification_report(ens_test, ens_pred))

    # %% Sarcasm, irony, figurative
    if 'si' in DS:

        # %% Data import stage
        fp = './sa/sample-data/sarcasm_irony_figurative_normal_tweets.csv'
        df = pd.read_csv(fp)
        df = df.loc[df['class'] != 'figurative']

        # %% Preprocessing
        df = run_preprocessing_pipeline(df)

    if RUN_MODELS and 'si' in DS:

        if 'nb' in MODELS:
            # %% Naive Bayes
            print('Running the Naive Bayes model for sarcasm/irony...')
            nb_test, nb_x_test, nb_pred, nb_model_si = naive_bayes_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(nb_test, nb_pred))
            joblib.dump(nb_model_si, 'nb_model_si.pkl')

        if 'lsvc' in MODELS:
            # %% Linear svc
            print('Running the Linear SVC model for sarcasm/irony...')
            lsvc_test, lsvc_x_test, lsvc_pred, lscv_model_si = linear_svc_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(lsvc_test, lsvc_pred))
            joblib.dump(nb_model_si, 'lsvc_model_si.pkl')

        if 'rf' in MODELS:
            # %% Random forest
            print('Running the Random Forest model...')
            rf_test, rf_x_test, rf_pred, rf_model_si = random_forest_mdl(df, log_level=LOG_LEVEL)
            print(classification_report(rf_test, rf_pred))
            joblib.dump(nb_model_si, 'rf_model_si.pkl')

    if RUN_ENSEMBLE and 'si' in DS:
        print('Running Ensemble Classifier for sarcasm/irony dataset...')
        models = [('nb', joblib.load('nb_model_si.pkl')), ('lsvc', joblib.load('lsvc_model_si.pkl'))]
        ens_test, ens_pred = ensemble_mdl(df, estimators=models)
        print(classification_report(ens_test, ens_pred))