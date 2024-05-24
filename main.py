import click
from sklearn.model_selection import train_test_split
import pickle
import main 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer  = WordNetLemmatizer()
en_stops = set(stopwords.words('english'))
tfidf_vectorizer = TfidfVectorizer()

# предобработка данных
def change_rating(row):
    if row > 3:
        return 3
    elif row < 3:
        return 1
    else:
        return 2
    
def text_prepocessing(row):
    row = re.sub(r"[^\w\s]", ' ', row.lower())
    row_list = row.split(' ')
    row_list_withut_stops = [word for word in row_list if word not in en_stops]
    text = [lemmatizer.lemmatize(w) for w in row_list_withut_stops]
    return ' '.join(text)

def data_preprocessing(data):
    data['turbo_clean_text'] = data['text'].apply(text_prepocessing)
    data['new_rating'] = data['rating'].apply(change_rating)
    return data


def my_train(data, test, split, model):
    #print('hi its 49')
    # Загрузить данные
    df = pd.read_csv('../data/' + data) # changeto ../data/
    #print('hi 52')
    data = data_preprocessing(df)
    #print('hi 54')
    data_text = data['turbo_clean_text']
    data_labels = data['new_rating']
    # Разделить данные на обучающие и тестовые (если указано)
    if split:
        #print('hi its 59')
        X_train, X_test, y_train, y_test = train_test_split(data_text, data_labels, test_size=split, random_state=42)
    if test:
        X_train = data_text
        y_train = data_labels
        #print('hi its 64')
        df_test = pd.read_csv('../data/' + test ) ## change to ../data/
        #print('hi its 66')
        df_test = data_preprocessing(df_test)
        df_test['turbo_clean_text'] = df_test['text'].apply(text_prepocessing) 
        X_test = df_test['turbo_clean_text']
        y_test =df_test['new_rating']
    else:
        X_train = data_text
        y_train = data_labels
    #print('hi its 73')
    # Обучить модель
    classifier = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            ("model", SVC(kernel='linear')),
        ]
    )
    classifier.fit(X_train, y_train)
    #print('hi its 83')
    # Сохранить модель
    
    with open('../data/' + model , 'w+b') as file:
        #print('helloooo')
        pickle.dump(classifier, file)


    # Оценить модель (если указаны тестовые данные)
    if X_test is not None and y_test is not None:
        y_pred = classifier.predict(X_test)
        accuracy = f1_score(y_test, y_pred, average='macro')
        print(f"F1-sscore модели: {accuracy}")


def my_predict(model, data):
    """Make predictions using a model."""
    if data.endswith('.csv'):
        df = (pd.read_csv('../data/' + data))
        X = df['text'] 
    else:
        X = [text_prepocessing(data)]
    #print('hi 101')
    with open('../data/' + model, 'rb') as f:
        #print('hiiii')
        great_model = pickle.load(f)
        #print('eto vtoroi')
        f.close()
        #print('104')
        
    
    predictions = great_model.predict(X)
  
    for i in predictions:
        print(i)

@click.command()
@click.option('--data', required=True, help='Path to the data')
@click.option('--test', help='Path to the test data (optional)')
@click.option('--split', type=float, help='Test set split ratio (optional)')
@click.option('--model', required=True, help='Path to the model')
@click.argument('command')
def main(command, data, test, split, model):
    if not data or not model:
        raise Exception('No data or model got')
    if command == 'train':
        if not os.path.isfile('/Users/anastasiakucina/project/ML_course/data/' + data):
            raise Exception('Data file does not exist')
        my_train(data, model, split, test)

    elif command == 'predict':
        my_predict(model, data)

    else:
        raise Exception("Invalid command")

if __name__ == '__main__':
    main()
