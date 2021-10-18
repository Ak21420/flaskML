from logging import exception
from flask import Flask, Blueprint, render_template, request, redirect, url_for, flash,Response
from flask_cors import CORS,cross_origin
import psycopg2
import psycopg2.extras
import numpy as np
from joblib import load
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from sklearn.feature_extraction.text import CountVectorizer


naive_bayes_algo = Blueprint("naive_bayes_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@naive_bayes_algo.route('/naive_bayes', methods=['GET', 'POST'])
@cross_origin()
def naive_bayes():
    try:
        model = load('joblib_files/naive_bayes.joblib')
        df = pd.read_csv("csv_files/spam.csv")
        df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
         
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'GET':
            try:
                return render_template('naive_bayes/naive.html')
            except Exception as e:
                print(e)
        else:
            message = request.form['message']

            message = message.replace("'", " ")

            message_list = [message]

            # print(message_list)
            X = df.Message
            y = df.spam

            v = CountVectorizer()
            X_count = v.fit_transform(X.values)

            message_count = v.transform(message_list)

            # print(message_count)
            # prediction = []
            try:
                prediction = model.predict(message_count)
            except Exception as e:
                print(e)
            
            Dtime = datetime.now()

            # print(prediction)

            # CREATE TABLE naive (
            #     NaiveID serial primary key NOT NULL,
            #     message text not null ,
            #     predict int not null,
            #     datetime text not null,
            #     is_delete bool
            # );

            try:
                cur.execute("INSERT INTO naive(message,predict,datetime,is_delete)VALUES (%s,%s,%s,FALSE)", (message,int(prediction[0]),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"
            
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM naive ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()

                return render_template('naive_bayes/naive_pred.html', naive_bayes = data)
            except Exception as e:
                print(e)

    except Exception as e:
        print(e)

@naive_bayes_algo.route('/naive_layout', methods=['GET', 'POST'])
def naive_layout():
    try:      
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM naive WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('naive_bayes/naive_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@naive_bayes_algo.route('/naive_delete/<string:id>', methods = ['POST','GET'])
def naive_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE naive SET is_delete = TRUE WHERE naiveid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('naive_bayes_algo.naive_layout'))
    except:
        return "Connection Fail"

@naive_bayes_algo.route('/naive_edit/<string:id>', methods = ['POST', 'GET'])
def naive_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM naive WHERE naiveid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()
        
        return render_template('naive_bayes/naive_edit.html', naive_bayes_data = data)
    except:
        return "Connection Fail"

@naive_bayes_algo.route('/naive_update/<string:id>', methods=['POST'])
def naive_update(id):
    if request.method == 'POST':
        message = request.form['message']

        message = message.replace("'", " ")
        
        model = load('joblib_files/naive_bayes.joblib')
        
        df = pd.read_csv("csv_files/spam.csv")
        df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        message_list = [message]

        # print(message_list)
        X = df.Message
        y = df.spam

        v = CountVectorizer()
        X_count = v.fit_transform(X.values)

        message_count = v.transform(message_list)

        try:
            prediction = model.predict(message_count)
        except Exception as e:
            print(e)
        
        Dtime = datetime.now()

        try:
            cur.execute("UPDATE naive SET message = %s, predict = %s, datetime = %s WHERE naiveid = %s", (message,int(prediction[0]),Dtime,id))
        except Exception as e:
            print(e)

        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute('SELECT * FROM naive ORDER BY datetime DESC')
            
            data = cur.fetchone()
            cur.close()

            return render_template('naive_bayes/naive_pred.html', naive_bayes = data)
        except Exception as e:
            print(e)