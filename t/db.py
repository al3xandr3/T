# -*- coding: utf-8 -*-
"""
Created on Tuesday July 13 2021

@author: amatos
"""

# References
# https://pythonspeed.com/articles/indexing-pandas-sqlite/
# http://al3xandr3.github.io/table-query.html

import pandas as pd
import sqlite3

class db:
    
    db_path    = ""
    table_name = ""
    csv_path   = ""
    _db = ""
    
    def __init__(self, db_path="C:/Users/amatos/data/database.sqlite", table_name="tabl"):

        self.db_path = db_path
        self.table_name = table_name
        try:
            self._db = sqlite3.connect(db_path)
            print("sqlite3.version: " + sqlite3.version)
        except Exception as e:
            print("Could not Connect to Database\n")
            print(e)

    def __del__(self):
        self._db.close()

    def execute(self, query):
        cur = self._db.cursor()
        cur.execute(query)

    def query(self, query):
        return pd.read_sql_query(query, self._db)

    def tables(self):
        return(self.query("select name from sqlite_master where type='table' ORDER BY name;"))

    def import_csv (self, csvfile_path, table_name="", index="", **kwargs):
        db = self._db
        table_name= table_name or self.table_name
        self.execute(f"drop table if exists {table_name};")

        # Load the CSV in chunks:
        for c in pd.read_csv(csvfile_path, chunksize=1000):
            # Append all rows to a new database table
            c.to_sql(table_name, db, if_exists="append")
        # Add an index on the 'street' column:
        #db.execute("CREATE INDEX street ON <table_name>(<index>)") 


if __name__ == "__main__":

    print("Run as a lib:")
    l = []
    for key, value in list(locals().items()):
        if callable(value) and value.__module__ == __name__:
            l.append(key)
    print(l)

    # --- USE ----
    # d = t.db()
    # path = "https://raw.githubusercontent.com/curran/data/gh-pages/oecd/houseprices-oecd-1975-2012.csv"
    # d.import_csv(path, "new_name")
    # d.tables()
    # d.execute("drop table if exists new_name;")
    # d.query(r"SELECT * FROM tabl LIMIT 5;")
    


