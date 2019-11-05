import sys
import nltk
nltk.download(['punkt', 'wordnet'])

from sklearn.externals import joblib
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    Load the data from the database. Assumes that the filepath has the same
    name as the table. (e.g. Database.db with table Database)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath[:-3].split('/')[-1], engine)
    X = df['message']
    Y = df[['related', 'request', 'offer',
           'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
           'security', 'military', 'child_alone', 'water', 'food', 'shelter',
           'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
           'infrastructure_related', 'transport', 'buildings', 'electricity',
           'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
           'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
           'other_weather', 'direct_report']]
    col_names = Y.columns
    return X, Y, col_names


def tokenize(text):
    """ Tokenizes and lemmatizes all text in a given string """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """ Builds a multioutput classifier with the pipeline module """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.8, min_df=0.0)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=2)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates the model """
    y_pred = model.predict(X_test)
    for i, col in enumerate(zip(Y_test.values.T, y_pred.T)):
        col_true, col_pred = col
        if not sum(pd.to_numeric(col_true)):
            print(print('Category: ', category_names[i]), ' is not present.')
            print('-'*50)
            continue
        print('Category: ', category_names[i])
        print('Number of related labels: ', sum(pd.to_numeric(col_true)))
        print(classification_report(col_true, col_pred))
        print('-'*50)


def save_model(model, model_filepath):
    """ Saves the model """
    with open(model_filepath, 'wb') as f:
        joblib.dump(model, f)


def main():
    """
    Example of use:
        python train_classifier.py location_of_sql_database location_to_save_model
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()