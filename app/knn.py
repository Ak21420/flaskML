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

knn_algo = Blueprint("knn_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@knn_algo.route('/knn', methods=['GET', 'POST'])
def knn():  
    try:
        model = load('Knn.joblib')
        data_set= pd.read_csv('csv_files/iris.csv') 

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('knn_folder/knn.html')
        else:
            x= data_set.iloc[:, 0:-1].values
            y= data_set.iloc[:, -1].values

            sepal_length = request.form['sepal_length']
            sepal_width = request.form['sepal_width']
            petal_length = request.form['petal_length']
            petal_width = request.form['petal_width']
            
            ar = []

            ar.append(sepal_length)
            ar.append(sepal_width)
            ar.append(petal_length)
            ar.append(petal_width)

            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values"

            Dtime = datetime.now()

            # CREATE TABLE knn (
            #     knnID serial primary key NOT NULL,
            #     sepal_length float not null ,
            #     sepal_width float not null ,
            #     petal_length float not null ,
            #     petal_width float not null ,
            #     predicted_value text not null ,
            #     datetime text not null,
            #     is_delete bool
            # );
            
            st_x = StandardScaler()

            st_x.fit(x)
            st_x.transform(x)

            ar_str = ','.join(ar)

            if ar:
                ar_new = floats_string_to_np_arr(ar_str) 
            else:
                return "Please enter value"
            
            x_real = np.array([ar_new[0],ar_new[1],ar_new[2],ar_new[3]]).reshape(1, -1)
            
            x_real_process = st_x.transform(x_real)  

            try:
                prediction = model.predict(pd.DataFrame(columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], data=x_real_process))
            except Exception as e:
                print(e)

            try:
                cur.execute("INSERT INTO knn(sepal_length,sepal_width,petal_length,petal_width,predicted_value,datetime,is_delete) VALUES (%s,%s,%s,%s,%s,%s,FALSE)", (str(float(ar[0])),str(float(ar[1])),str(float(ar[2])),str(float(ar[3])),str(prediction[0]),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if len(ar):
                
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM knn ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()
                print(data)
                request_type_str = request.method

                return render_template('knn_folder/knn_pred.html', knn = data)
            else:
                return "Please enter Valid values"
    except Exception as e:
        print(e)


@knn_algo.route('/knn_layout', methods=['GET', 'POST'])
def knn_layout():
    try:
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM knn WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('knn_folder/knn_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@knn_algo.route('/knn_delete/<string:id>', methods = ['POST','GET'])
def knn_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE knn SET is_delete = TRUE WHERE knnid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('knn_algo.knn_layout'))
    except:
        return "Connection Fail"

@knn_algo.route('/knn_edit/<string:id>', methods = ['POST', 'GET'])
def knn_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM knn WHERE knnid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('knn_folder/knn_edit.html', knn = data)
    except:
        return "Connection Fail"

@knn_algo.route('/knn_update/<string:id>', methods=['POST'])
def knn_update(id):
    if request.method == 'POST':      
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':    
            model = load('Knn.joblib')
            data_set= pd.read_csv('csv_files/iris.csv') 
         
            x= data_set.iloc[:, 0:-1].values
            y= data_set.iloc[:, -1].values
         
            sepal_length = request.form['sepal_length']
            sepal_width = request.form['sepal_width']
            petal_length = request.form['petal_length']
            petal_width = request.form['petal_width']
            
            ar = []
         
            ar.append(sepal_length)
            ar.append(sepal_width)
            ar.append(petal_length)
            ar.append(petal_width)

            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values"

            Dtime = datetime.now()
    
            st_x = StandardScaler()

            st_x.fit(x)
            st_x.transform(x)

            ar_str = ','.join(ar)
         
            if ar:
                ar_new = floats_string_to_np_arr(ar_str) 
            else:
                return "Please enter value"
            
            x_real = np.array([ar_new[0],ar_new[1],ar_new[2],ar_new[3]]).reshape(1, -1)
            
            x_real_process = st_x.transform(x_real)  

            try:
                prediction = model.predict(pd.DataFrame(columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'], data=x_real_process))
            except Exception as e:
                print(e)

            try:
                cur.execute("UPDATE knn SET sepal_length = %s, sepal_width = %s, petal_length = %s, petal_width = %s, predicted_value = %s, datetime = %s WHERE knnid = %s", (str(float(ar[0])),str(float(ar[1])),str(float(ar[2])),str(float(ar[3])),str(prediction[0]),Dtime,id))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if len(ar):
                
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM knn ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()
                print(data)
                request_type_str = request.method

                return render_template('knn_folder/knn_pred.html', knn = data)
            else:
                return "Please enter Valid values"


def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            if float(s) >= 0:
                return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)

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