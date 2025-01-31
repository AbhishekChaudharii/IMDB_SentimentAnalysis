import pandas as pd
import numpy as np
import sqlite3
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from string import punctuation
import os



def data_prep():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Data", "IMDB_Dataset.csv")
    df = pd.read_csv(file_path)
    df['sentiment'] = df['sentiment'].apply(lambda x:1 if x=='positive' else 0)
    return df

def data_cleaning(text):
    text = text.lower()
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    text = re.sub('\[[^]]*\]', '', text)
    return text

def remove_stopwords(text,punctuation=punctuation):
    stop = set(stopwords.words('english'))
    punctuation = list(punctuation)
    stop.update(punctuation)
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)

def db_setup():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('imdb_reviews.db')
    cursor = conn.cursor()
    # Create the table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT,
        sentiment INTEGER
    )
    ''')

    # Commit the changes
    conn.commit()
    return conn

def db_store(df,conn):
    """Store preprocessed data in SQLite database"""
    # Insert data into the table
    df.to_sql('imdb_reviews', conn, if_exists='replace')
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()



if __name__ == "__main__":
    print("Importing Data")
    df = data_prep()
    print("Processing Data")
    df['review']=df['review'].apply(data_cleaning)
    df['review']=df['review'].apply(remove_stopwords)
    print("Setting up SQLite DB...")
    conn = db_setup()
    print("Loading into database")
    db_store(df,conn)
