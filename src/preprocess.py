import nltk
import spacy
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def _set_environment():
    '''
    Function to set the environment for the preprocess_text_and_store function
    '''
    try:
        nlp = spacy.load('en_core_web_sm')
    except:
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    try:
        eng_stopwords = nltk.corpus.stopwords.words('english')
    except:
        nltk.download('stopwords')
        eng_stopwords = nltk.corpus.stopwords.words('english')
    try:
        _ = nltk.tokenize.word_tokenize('test')
    except:
        nltk.download('punkt')
    return nlp, eng_stopwords

def preprocess_text_and_store(text:np.array, doc_store:str=None, store:bool=False):
    '''
    Function to preprocess the text
    Parameters:
        text: np.array, text to preprocess
        doc_store: string, name of the file where to store the preprocessed text
        store: bool, if True store the preprocessed text
    '''
    nlp, eng_stopwords = _set_environment()
    preprocessed_text = np.empty(len(text), dtype=object)
    if doc_store is None or doc_store not in os.listdir('../data'):
        counter = 0
        for i in range(len(text)):
            process_words = []
            text[i] = text[i].replace('\d', ' ')
            for word in nltk.word_tokenize(nlp(text[i].lower()).text):
                if word.isalpha() and word not in eng_stopwords and len(str(word)) >= 3 and word != 'subject':
                    process_words.append(word)
            preprocessed_text[counter] = ' '.join(process_words)
            counter += 1
    else:
        preprocessed_text = np.array(pd.read_csv('../data/'+doc_store)['comment_text'])
    if store:
        pd.DataFrame(data = {"comment_text":preprocessed_text}).to_csv('../data/'+doc_store)

    return preprocessed_text

def train_test_val_split(data:np.array, labels:np.array=None, test_size:float=0.2, val_size:float=0.1, random_state:int=42):
    '''
    Function to split the data into train, test and validation sets
    Parameters:
        data: np.array, data to split
        labels: np.array, labels to split
        test_size: float, size of the test set
        val_size: float, size of the validation set
        random_state: int, random state
    '''
    if labels is None:
        data, labels = data
    corpus, X_test, train_labels, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(corpus, train_labels, test_size=val_size/(1-test_size), random_state=random_state)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_val = np.array(y_val)

    return X_train.astype(str).tolist(), X_test.astype(str).tolist(), X_val.astype(str).tolist(), y_train, y_test, y_val
