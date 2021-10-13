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

logistic_algo = Blueprint("logistic_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@logistic_algo.route('/logistic', methods=['GET', 'POST'])
def logistic():   
    try:
        model = load('joblib_files/logistic.joblib')
        data_set= pd.read_csv('csv_files/Social_Network_Ads.csv') 

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('logistic_folder/logistic.html')
        else:
            x= data_set.iloc[:, [0,1]].values
            y= data_set.iloc[:, 2].values

            age = request.form['age']
            expected_salary = request.form['expected_salary']
            
            ar = [] 

            ar.append(age)
            ar.append(expected_salary)

            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values"

                if float(age) > 100:
                    return "Write Posible values"

            Dtime = datetime.now()

            # CREATE TABLE logistic (
            #     LogisticID serial primary key NOT NULL,
            #     age text not null ,
            #     expected_salary text not null ,
            #     predicted_value text not null ,
            #     datetime text not null,
            #     is_delete bool
            # );

            st_x = StandardScaler()

            st_x.fit(x)
            st_x.transform(x)

            if age:
                age = floats_string_to_np_arr(age) 
            else:
                return "Please enter value"
            
            if expected_salary:
                expected_salary = floats_string_to_np_arr(expected_salary) 
            else:
                return "Please enter value"

            x_real = np.array([age,expected_salary]).reshape(1, -1)
            x_real_process = st_x.transform(x_real)  

            try:
                prediction = model.predict(pd.DataFrame(columns=['age', 'expected_salary'], data=x_real_process))
            except Exception as e:
                print(e)

            try:
                cur.execute("INSERT INTO logistic(age,expected_salary,predicted_value,datetime,is_delete) VALUES (%s,%s,%s,%s,FALSE)", (str(int(age[0][0])),str(int(expected_salary[0][0])),str(prediction[0]),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if len(age) & len(expected_salary):
                
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM logistic ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()
                print(data)
                request_type_str = request.method

                return render_template('logistic_folder/logistic_pred.html', logistic = data)
            else:
                return "Please enter Valid values"
    except Exception as e:
        print(e)

@logistic_algo.route('/logistic_layout', methods=['GET', 'POST'])
def logistic_layout():
    try: 
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM logistic WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('logistic_folder/logistic_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@logistic_algo.route('/logistic_delete/<string:id>', methods = ['POST','GET'])
def logistic_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE logistic SET is_delete = TRUE WHERE logisticid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('logistic_algo.logistic_layout'))
    except:
        return "Connection Fail"

@logistic_algo.route('/logistic_edit/<string:id>', methods = ['POST', 'GET'])
def logistic_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM logistic WHERE logisticid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('logistic_folder/logistic_edit.html', logistic = data)
    except:
        return "Connection Fail"

@logistic_algo.route('/logistic_update/<string:id>', methods=['POST'])
def logistic_update(id):
    if request.method == 'POST':  
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':    
            model = load('joblib_files/logistic.joblib')
            data_set= pd.read_csv('csv_files/Social_Network_Ads.csv') 

            x= data_set.iloc[:, [0,1]].values
            y= data_set.iloc[:, 2].values

            age = request.form['age']
            expected_salary = request.form['expected_salary']

            ar = [] 

            ar.append(age)
            ar.append(expected_salary)

            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values"

                if float(age) > 100:
                    return "Write Posible values"

            Dtime = datetime.now()

            st_x = StandardScaler()

            st_x.fit(x)
            st_x.transform(x)

            if age:
                age = floats_string_to_np_arr(age) 
            else:
                return "Please enter value"
            
            if expected_salary:
                expected_salary = floats_string_to_np_arr(expected_salary) 
            else:
                return "Please enter value"

            x_real = np.array([age,expected_salary]).reshape(1, -1)
            
            x_real_process = st_x.transform(x_real)

            try:
                prediction = model.predict(pd.DataFrame(columns=['age', 'expected_salary'], data=x_real_process))
            except Exception as e:
                print(e)

            try:
                cur.execute("UPDATE logistic SET age = %s, expected_salary = %s, predicted_value = %s, datetime = %s WHERE logisticid = %s", (str(int(age[0][0])),str(int(expected_salary[0][0])),str(prediction[0]),Dtime,id))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if len(age) & len(expected_salary):
                
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)   
                cur.execute('SELECT * FROM logistic ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()
                print(data)
                request_type_str = request.method

                return render_template('logistic_folder/logistic_pred.html', logistic = data)
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