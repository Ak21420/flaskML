from logging import exception
import re
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
import uuid
from datetime import date, datetime
from sklearn.preprocessing import StandardScaler

kmeans_algo = Blueprint("kmeans_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@kmeans_algo.route('/kmeans', methods=['GET', 'POST'])
def kmeans():   
    try:
        model = load('joblib_files/kmeans.joblib')
       
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('kmeans_folder/kmeans.html')
        else:
            age = request.form['age']
            income = request.form['income']
            spending_score = request.form['spending_score']
            
            ar = [] 
            ar.append(age)
            ar.append(income)
            ar.append(spending_score)

            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values"

                if float(age) > 100:
                    return "Write Posible value in Age"

                if float(spending_score) > 100 or float(spending_score) < 1:
                    return "Write Posible value in Spending Score"

            Dtime = datetime.now()

            # CREATE TABLE kmeans (
            #     KmeansID serial primary key NOT NULL,
            #     age int not null ,
            #     income int not null ,
            #     spending_score int not null, 
            #     predicted_value text not null ,
            #     datetime text not null,
            #     is_delete bool
            # );

            x_real = np.array([int(age),int(income),int(spending_score)]).reshape(1, -1)

            try:
                prediction = model.predict(pd.DataFrame(columns=['age', 'income', 'spending_score'], data = x_real))
            except Exception as e:
                print(e)

            prediction[0] = int(prediction[0]) + 1

            if int(prediction[0]) == 1:
                predict = 'General'
            elif int(prediction[0]) == 2:
                predict = 'Spend Thrift'
            elif int(prediction[0]) == 3:
                predict = 'Target'
            elif int(prediction[0]) == 4:
                predict = 'Miser'
            elif int(prediction[0]) == 5:
                predict = 'Careful'

            try:
                cur.execute("INSERT INTO kmeans(age,income,spending_score,predicted_value,datetime,is_delete) VALUES (%s,%s,%s,%s,%s,FALSE)", (int(age),int(income),int(spending_score),str(predict),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
            cur.execute('SELECT * FROM kmeans ORDER BY datetime DESC')
                
            data = cur.fetchone()
            cur.close()
            print(data)
            request_type_str = request.method

            return render_template('kmeans_folder/kmeans_pred.html', kmeans = data)
            
    except Exception as e:
        print(e)

@kmeans_algo.route('/kmeans_layout', methods=['GET', 'POST'])
def kmeans_layout():
    try: 
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM kmeans WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('kmeans_folder/kmeans_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@kmeans_algo.route('/kmeans_delete/<string:id>', methods = ['POST','GET'])
def kmeans_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE kmeans SET is_delete = TRUE WHERE kmeansid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('kmeans_algo.kmeans_layout'))
    except:
        return "Connection Fail"

@kmeans_algo.route('/kmeans_edit/<string:id>', methods = ['POST', 'GET'])
def kmeans_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM kmeans WHERE kmeansid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('kmeans_folder/kmeans_edit.html', kmeans = data)
    except:
        return "Connection Fail"

@kmeans_algo.route('/kmeans_update/<string:id>', methods=['POST'])
def kmeans_update(id):
    if request.method == 'POST':  
        model = load('joblib_files/kmeans.joblib')
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        age = request.form['age']
        income = request.form['income']
        spending_score = request.form['spending_score']
        
        ar = [] 
        ar.append(age)
        ar.append(income)
        ar.append(spending_score)

        if ar:
            st = check_values(ar)

            if st=="Success":
                pass
            else:
                return "Write Posible values"

            if float(age) > 100:
                return "Write Posible value in Age"

            if float(spending_score) > 100 or float(spending_score) < 1:
                return "Write Posible value in Spending Score"

        Dtime = datetime.now()

        x_real = np.array([int(age),int(income),int(spending_score)]).reshape(1, -1)

        try:
            prediction = model.predict(pd.DataFrame(columns=['age', 'income', 'spending_score'], data = x_real))
        except Exception as e:
            print(e)

        prediction[0] = int(prediction[0]) + 1

        if int(prediction[0]) == 1:
            predict = 'General'
        elif int(prediction[0]) == 2:
            predict = 'Spend Thrift'
        elif int(prediction[0]) == 3:
            predict = 'Target'
        elif int(prediction[0]) == 4:
            predict = 'Miser'
        elif int(prediction[0]) == 5:
            predict = 'Careful'

        try:
            cur.execute("UPDATE kmeans SET age = %s, income = %s, spending_score = %s, predicted_value = %s, datetime = %s WHERE kmeansid = %s", (int(age),int(income),int(spending_score),str(predict),Dtime,id))
        except Exception as e:
            print(e)

        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute('SELECT * FROM kmeans ORDER BY datetime DESC')
            
        data = cur.fetchone()
        cur.close()
        print(data)
        request_type_str = request.method

        return render_template('kmeans_folder/kmeans_pred.html', kmeans = data)


def check_values(values_array):
    ar = []
    for i in values_array:
        if float(i):
            if float(i)>=0:
                pass
            else:
                return "Fail"
        else:
            return "Fail"

    return "Success"