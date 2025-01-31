import sqlite3
conn = sqlite3.connect('imdb_reviews.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM imdb_reviews LIMIT 5')
rows = cursor.fetchall()

for idx,r,s in rows:
    print(r)
    print(s)

conn.close()