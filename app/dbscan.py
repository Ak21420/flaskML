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
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from pandas import DataFrame

dbscan_algo = Blueprint("dbscan_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@dbscan_algo.route('/dbscan', methods=['GET', 'POST'])
def dbscan():   
    try:
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('dbscan_folder/dbscan.html')
        else:
            number1 = request.form['number1']
            number2 = request.form['number2']
            
            ar = [] 
            ar.append(number1)
            ar.append(number2)

            if ar:
                st = check_values(ar)

                if st=="Fail":
                    return "Write Posible values"

                if float(number1) < -10 or float(number1) > 10:
                    return "Write Posible values"

                if float(number2) < -10 or float(number2) > 10:
                    return "Write Posible values"

            Dtime = datetime.now()

            # CREATE TABLE dbscan (
            #     DbscanID serial primary key NOT NULL,
            #     number1 text not null ,
            #     number2 text not null ,
            #     predicted_value text not null ,
            #     datetime text not null,
            #     is_delete bool
            # );

            try:
                prediction = makemodel(ar)
            except Exception as e:
                print(e)

            if str(prediction) == '-1':
                pred = "Noise"
            else:
                pred = "Cluster" + str(prediction)
            
            try:
                cur.execute("INSERT INTO dbscan(number1,number2,predicted_value,datetime,is_delete) VALUES (%s,%s,%s,%s,FALSE)", (str(number1),str(number2),str(pred),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute('SELECT * FROM dbscan ORDER BY datetime DESC')
            
            data = cur.fetchone()
            cur.close()
            print(data)

            return render_template('dbscan_folder/dbscan_pred.html', dbscan = data)
            
    except Exception as e:
        print(e)

@dbscan_algo.route('/dbscan_layout', methods=['GET', 'POST'])
def dbscan_layout():
    try: 
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM dbscan WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('dbscan_folder/dbscan_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@dbscan_algo.route('/dbscan_delete/<string:id>', methods = ['POST','GET'])
def dbscan_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE dbscan SET is_delete = TRUE WHERE dbscanid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('dbscan_algo.dbscan_layout'))
    except:
        return "Connection Fail"

@dbscan_algo.route('/dbscan_edit/<string:id>', methods = ['POST', 'GET'])
def dbscan_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM dbscan WHERE dbscanid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('dbscan_folder/dbscan_edit.html', dbscan = data)
    except:
        return "Connection Fail"

@dbscan_algo.route('/dbscan_update/<string:id>', methods=['POST'])
def dbscan_update(id):
    if request.method == 'POST':  
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        number1 = request.form['number1']
        number2 = request.form['number2']
        
        ar = [] 
        ar.append(number1)
        ar.append(number2)

        if ar:
            st = check_values(ar)

            if st=="Fail":
                return "Write Posible values"

            if float(number1) < -10 or float(number1) > 10:
                return "Write Posible values"

            if float(number2) < -10 or float(number2) > 10:
                return "Write Posible values"

        Dtime = datetime.now()

        try:
            prediction = makemodel(ar)
        except Exception as e:
            print(e)

        if str(prediction) == '-1':
            pred = "Noise"
        else:
            pred = "Cluster" + str(prediction)
        
        try:
            cur.execute("UPDATE dbscan SET number1 = %s, number2 = %s, predicted_value = %s, datetime = %s WHERE dbscanid = %s", (str(number1),str(number2),str(pred),Dtime,id))
        except Exception as e:
            print(e)

        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute('SELECT * FROM dbscan ORDER BY datetime DESC')
        
        data = cur.fetchone()
        cur.close()
        print(data)

        return render_template('dbscan_folder/dbscan_pred.html', dbscan = data)


def check_values(values_array):
    ar = []
    for i in values_array:
        if float(i):
            pass
        else:
            return "Fail"

    return "Success"


def makemodel(x_new):
    
    data_set, _= make_blobs(n_samples = 500, centers = 3, n_features = 2, center_box=(-10.0, 10.0),random_state = 20)

    # df = DataFrame(dict(x = data_set[:,0], y = data_set[:,1]))

    combined_array = np.append(data_set, x_new)
    combined_array = combined_array.reshape(501,2)
    
    clustering = DBSCAN(eps = 1, min_samples = 5).fit(combined_array)

    if clustering.labels_[-1] == -1:
        return clustering.labels_[-1]
    else:
        return clustering.labels_[-1]+1

