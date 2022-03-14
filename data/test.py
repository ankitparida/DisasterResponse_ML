import sqlite3
import pandas as pd

#Querying the Disaster table using pandas read_sql method

conn = sqlite3.connect('DisasterResponse.db')

print(pd.read_sql('SELECT * from DisasterResponse.db',conn))

conn.close()