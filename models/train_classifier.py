# import libraries
import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    '''
    Input:
    database_filename: the path of db
    Output:
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    print(engine.table_names())
    # read from db
    df = pd.read_sql_table('RESPONSES', con=engine)
    
    # slipt in input X and output Y
    X = df.message
    Y = df.loc[:, 'related':'direct_report']
    
    category_names=list(df.columns[4:])# categories
    
    return X, Y, category_names


def tokenize(text):
    '''
    Input:
        text: a  message
    Output:
        text_stemmed: cleaned message
    '''    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # lematize & remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text_lemmed = [lemmatizer.lemmatize(word, pos='v') for word in tokens if word not in stop_words]
    
    # Find stems
    # stemmer = PorterStemmer()
    # text_stemmed = [stemmer.stem(word) for word in text_lemmed]

    return text_lemmed


def build_model():
    '''
    Input:
    Output:
    cv : Gridsearch
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
            ])
    
    # find the best parameters
    parameters = {'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[50, 100, 200, 250],
                  'clf__estimator__learning_rate':[0.5,1.0,1.5,2]}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input
    model: model to evaluate
    X_test: Input test data
    Y_test: True output data
    category_name: Labels for all categories
    Output:
    Print Classification report
    '''
    y_pred = model.predict(X_test)
    #print classification report for every category
    for i,col in enumerate(Y.columns):
        print(f"###################   {col}   ####################")
        print(classification_report(list(Y_test.values[:, i]), list(y_pred[:, i])))


def save_model(model, model_filepath):
    '''
    Input:
        model:model for save
        model_filepath: where to save
    '''
    
    pickle.dump(model, open(model_filepath, "wb"))


def main():
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