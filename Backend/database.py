import pymysql

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="password",   # change this
        database="career_platform",
        cursorclass=pymysql.cursors.DictCursor
    )
