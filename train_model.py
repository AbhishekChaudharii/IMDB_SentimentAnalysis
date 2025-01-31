from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
import numpy as np
import sqlite3

def load_data():
    """Load data from SQLite database"""
    conn = sqlite3.connect('imdb_reviews.db')
    df = pd.read_sql_query("SELECT review,sentiment FROM imdb_reviews", conn)
    conn.close()
    return df

def train_model(df,model):
    """df:DataFrame
    model: str
        Two options for model
        lr:Logistic Regression
        nb:Naive Bayes
    """
    x_train,x_test,y_train,y_test = train_test_split(df.review,df.sentiment)
    tfv=TfidfVectorizer(use_idf=True,ngram_range=(1,3))
    #transformed train reviews
    tfv_train_reviews=tfv.fit_transform(x_train)
    #transformed test reviews
    tfv_test_reviews=tfv.transform(x_test)
    #training the model
    lr=LogisticRegression(penalty='l2',max_iter=500,random_state=42)
    nb = MultinomialNB()
    #Fitting the models for tfidf features
    nb_tfidf = nb.fit(tfv_train_reviews,y_train)
    lr_tfidf=lr.fit(tfv_train_reviews,y_train)
    lr_tfidf_predict=lr.predict(tfv_test_reviews)
    nb_tfidf_predict = nb.predict(tfv_test_reviews)
    #Classification report for tfidf features
    lr_tfidf_report=classification_report(y_test,lr_tfidf_predict)
    nb_tfidf_report=classification_report(y_test,nb_tfidf_predict,)
    print("\n Logistic Regression",lr_tfidf_report)
    print("\n Naive Bayes",nb_tfidf_report)
    if model=='lr':
        return lr,tfv
    else:
        return nb,tfv

def export_model(model,vectorizer):
    print("\nSaving model and vectorizer...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)






if __name__ == "__main__":
    df= load_data()
    lr,tfv = train_model(df,'lr')
    export_model(lr,tfv)
    



