import sys
import sqlite3
import pandas as pd
import nltk
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import sqlite3

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''
    This function does the following:
    a.load the data into dataframe from database
    b.extract the message column as Feature/Independent Variable
    c.extract the category columns as Targets/Dependent Variable
    d.return feature, targets/labels, and category names
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable',con=engine)
    
    #df.head()
    X = df['message'] 
    Y = df.iloc[:,4:]
    target_names = Y.columns
    
    return X,Y,target_names


def tokenize(text):
    '''
    This function tokenize the given word and return the individual tokens
    
    INPUT:
    text: String which will be divided into tokens
    
    OUTPUT:
    tokens: Individual Tokens after performing Lemmatization and removing the stop words
    '''
    normalized_text = re.sub(r'[^a-zA-Z0-9]',' ',str(text)).lower()
    word_tokens = word_tokenize(normalized_text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(word) for word in word_tokens  if word not in stop_words]
    return tokens


def build_model():
    '''
    This function does the following:
    a.Create a Pipeline
    b.Defining the Parameters required for model building
    c.Create and returns a CV object
    '''
    # text processing and model pipeline
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer = tokenize)),
        ('tfidf',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
         
    ])
    
    # define parameters for GridSearchCV
    parameters = {
    'clf__estimator__n_estimators': [5,10]
    }
    
    # create gridsearch object and return as final model pipeline
    cv = GridSearchCV(pipeline,param_grid=parameters,cv = 3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function does the following:
    - use the provided machine learning pipeline to predict categories of messages
    - print out the model performance evaluation results
    '''
    
    # output model test results
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,target_names=category_names))
 
        
def save_model(model, model_filepath):
    '''
    This Function save the machine learning model using Pickle Library
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    


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