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

sLinear = Blueprint("sLinear", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@sLinear.route('/simple_linear', methods=['GET', 'POST'])
def simple_linear():
    
    try:
        request_type_str = request.method      
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            print("test")
            return render_template('simple_linear_folder/sLinear.html')
        else:
            text = request.form['text']

            Dtime = datetime.now()

            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            model = load('model.joblib')

            try:
                cur.execute("INSERT INTO simple(txt_numbers,datetime,image,is_delete) VALUES (%s,%s,%s,FALSE)", (text,Dtime,random_string))
            except:
                return "Third"

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if text:
                np_arr = floats_string_to_np_arr(text) 
            else:
                return "Please enter value"
            print("test")
            if len(np_arr):
                make_picture('AgesAndHeights.pkl', model, np_arr, path)
                return render_template('simple_linear_folder/sLinear.html', href=path)
            else:
                return "Please enter Valid values => " + text
    except Exception as e:
        print(e)

@sLinear.route('/simple_linear_layout', methods=['GET', 'POST'])
def simple_linear_layout():
    try:
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM simple WHERE is_delete=FALSE"
        cur.execute(s)
        list_users = cur.fetchall()
        cur.close()

        return render_template('simple_linear_folder/sLinear_layout.html', list_users = list_users)
    except:
        return "Connection Fail"

@sLinear.route('/simple_Linear_delete/<string:id>', methods = ['POST','GET'])
def simple_Linear_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE simple SET is_delete = TRUE WHERE simpleid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('sLinear.simple_linear_layout'))
    except Exception as e:
        print(e)

@sLinear.route('/simple_Linear_edit/<string:id>', methods = ['POST', 'GET'])
def simple_Linear_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM simple WHERE simpleid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()
        
        return render_template('simple_linear_folder/sLinear_edit.html', sLinear = data)
    except Exception as e:
        print(e)

@sLinear.route('/simple_Linear_update/<string:id>', methods=['POST'])
def simple_Linear_update(id):
    if request.method == 'POST':
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':            
            text = request.form['txt_numbers']

            Dtime = datetime.now()

            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            model = load('model.joblib')

            try:
                cur.execute("UPDATE simple SET txt_numbers = %s, image = %s, datetime = %s WHERE simpleid = %s", (text, random_string, Dtime, id))
            except:
                return "Third"

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            if text:
                np_arr = floats_string_to_np_arr(text) 
            else:
                return "Please enter value"

            if len(np_arr):
                make_picture('AgesAndHeights.pkl', model, np_arr, path)
                return redirect(url_for('sLinear.simple_linear_layout'))
            else:
                return "Please enter Valid values => " + text


def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    data = data[ages > 0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.array(list(range(19))).reshape(19, 1)
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Age of People", labels={'x': 'Age (years)',
                        'y': 'Height (inches)'})

    fig.add_trace(go.Scatter(x=x_new.reshape(19), y=preds, mode='lines', name='Model'))

    new_preds = model.predict(new_inp_np_arr)

    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, name='New Outputs', mode='markers', marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))
    
    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()


def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            if float(s) >= 0:
                return True
        except:
            return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)
