from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg2
import psycopg2.extras
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime

app = Flask(__name__, static_url_path = "", static_folder = "static", template_folder = "templates")
app.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "simple_linear"
DB_USER = "admin"
DB_PASS = "admin"


@app.route('/')
def home():
   return render_template('index.html')

# @app.route('/')
# def Index():
#     cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
#     s = "SELECT * FROM simple"
#     cur.execute(s) # Execute the SQL
#     list_users = cur.fetchall()
#     return render_template('simple_linear.html', list_users = list_users)


# @app.route('/simple_linear_add', methods=['GET','POST'])
# def add_student():
    # cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # if request.method == 'POST':
    #     fname = request.form['fname']
    #     lname = request.form['lname']
    #     email = request.form['email']
        # cur.execute("INSERT INTO students (fname, lname, email) VALUES (%s,%s,%s)", (fname, lname, email))
        # conn.commit()
        # flash('Student Added successfully')
        # return redirect(url_for('Index'))
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

@app.route('/simple_linear', methods=['GET', 'POST'])
def simple_linear():
    # print("Simple_Linear")
    try:
        # print("First")
        
        # print("second")
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        # cur.execute("SELECT image FROM simple ORDER BY datetime DESC")
        # image_link = cur.fetchone()
        
        # image_name = "static/" + image_link[0] + ".svg"
        
        # print(image_name)

        if request_type_str == 'GET':
            # static/37856fd1878145a084368d92a75424c7.svg
            # print("2131")
            return render_template('sLinear.html')
        else:
            # print("Third")
            text = request.form['text']

            Dtime = datetime.now()

            # 6a0dcc9c03574ae0b563ea938316d37f
            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            model = load('model.joblib')

            try:
                cur.execute("INSERT INTO simple(txt_numbers,datetime,image) VALUES (%s,%s,%s)", (text,Dtime,random_string))
            except:
                return "Third"

            try:
                conn.commit()
            except:
                return "Forth"

            # print("Forth")
            # flash('Data Added Successfully!!!')
            # return redirect(url_for('Index'))
            # print("Fifth")            

            if text:
                np_arr = floats_string_to_np_arr(text) 
            else:
                return "Please enter value"

            if len(np_arr):
                make_picture('AgesAndHeights.pkl', model, np_arr, path)
                return render_template('sLinear.html', href=path)
            else:
                return "Please enter Valid values => " + text

    except:
        return "Connection Fail"


@app.route('/simple_linear_layout', methods=['GET', 'POST'])
def simple_linear_layout():
    # print("Simple_Linear")
    try:
        
        # print("second")
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        # print("First")
            
        s = "SELECT * FROM simple"
        cur.execute(s) # Execute the SQL
        list_users = cur.fetchall()
      
        # print("2131")
        # print(list_users)
        return render_template('sLinear_layout.html', list_users = list_users)

    except:
        return "Connection Fail"



@app.route('/delete/<string:id>', methods = ['POST','GET'])
def delete_student(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    cur.execute('DELETE FROM simple WHERE simpleid = {0}'.format(id))
    conn.commit()
    flash('Data Removed Successfully')
    return redirect(url_for('simple_linear_layout'))


@app.route('/edit/<id>', methods = ['POST', 'GET'])
def get_employee(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
   
    cur.execute('SELECT * FROM simple WHERE simpleid = %s', (id))
    data = cur.fetchall()
    cur.close()
    print(data[0])
    return render_template('sLinear_edit.html', sLinear = data[0])



@app.route('/update/<id>', methods=['POST'])
def update_student(id):
    if request.method == 'POST':
        

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        
        if request_type_str == 'POST':
            # print("Third")
            text = request.form['txt_numbers']

            Dtime = datetime.now()

            # 6a0dcc9c03574ae0b563ea938316d37f

            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            model = load('model.joblib')

            try:
                cur.execute("UPDATE simple SET txt_numbers = %s, image = %s, datetime = %s WHERE simpleid = %s", (text, random_string, Dtime, id))
            except:
                return "Third"

            try:
                conn.commit()
            except:
                return "Forth"

            # print("Forth")
            # flash('Data Added Successfully!!!')
            # return redirect(url_for('Index'))
            # print("Fifth")            

            if text:
                np_arr = floats_string_to_np_arr(text) 
            else:
                return "Please enter value"

            if len(np_arr):
                make_picture('AgesAndHeights.pkl', model, np_arr, path)
                return redirect(url_for('simple_linear_layout'))
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