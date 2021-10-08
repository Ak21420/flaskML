from flask import Flask, render_template, request, redirect, url_for, flash
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from datetime import datetime


app = Flask(__name__, static_url_path = "", static_folder = "static", template_folder = "templates")
app.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"


@app.route('/')
def home():
   return render_template('index.html')

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)

@app.route('/simple_linear', methods=['GET', 'POST'])
def simple_linear():
    
    try:
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('sLinear.html')
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

            if len(np_arr):
                make_picture('AgesAndHeights.pkl', model, np_arr, path)
                return render_template('sLinear.html', href=path)
            else:
                return "Please enter Valid values => " + text

    except:
        return "Connection Fail"


@app.route('/simple_linear_layout', methods=['GET', 'POST'])
def simple_linear_layout():
    try:
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM simple WHERE is_delete=FALSE"
        cur.execute(s) # Execute the SQL
        list_users = cur.fetchall()
        cur.close()

        return render_template('sLinear_layout.html', list_users = list_users)

    except:
        return "Connection Fail"

@app.route('/simple_Linear_delete/<string:id>', methods = ['POST','GET'])
def simple_Linear_delete(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute('UPDATE simple SET is_delete = TRUE WHERE simpleid = {0}'.format(id))
    conn.commit()
    cur.close()
    
    return redirect(url_for('simple_linear_layout'))

@app.route('/simple_Linear_edit/<string:id>', methods = ['POST', 'GET'])
def simple_Linear_edit(id):
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    cur.execute('SELECT * FROM simple WHERE simpleid = %s', (id,))
    
    data = cur.fetchone()
    cur.close()
    
    return render_template('sLinear_edit.html', sLinear = data)



@app.route('/simple_Linear_update/<string:id>', methods=['POST'])
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


@app.route('/multi_linear', methods=['GET', 'POST'])
@cross_origin()
def multi_linear():
    # print("Simple_Linear")
    try:
        print("First")

        model=pickle.load(open('mLinear.pkl','rb'))

        print("Second")

        car=pd.read_csv('Cleaned_Car_data.csv')

        print("Third")
        
        request_type_str = request.method
        
        try:
            print("forth")
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        print("Fifth")

        if request_type_str == 'GET':
            companies = sorted(car['company'].unique())
            car_models = sorted(car['name'].unique())
            year = sorted(car['year'].unique(),reverse=True)
            fuel_type = car['fuel_type'].unique()
            # print(companies)
            # print(car_models)
            # print(year)
            # print(fuel_type)
            companies.insert(0,'Select Company')
            print("Sixth")
            return render_template('mLinear.html',companies = companies, car_models = car_models, years = year,fuel_types = fuel_type)
        else:
            # sepal_length = request.form['sepal_length']
            # sepal_width = request.form['sepal_width']
            # petal_length = request.form['petal_length']
            # petal_width = request.form['petal_width']
            # mae  = request.form['mae']
            # mse = request.form['mse']
            # rmse = request.form['rmse']
            print("First")
            company_name=request.form.get('company')

            car_model=request.form.get('car_models')
            purchase_year=request.form.get('year')
            fuel_type=request.form.get('fuel_type')
            driven=request.form.get('kilo_driven')
            print("two")
            
            prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                    data=np.array([car_model,company_name,purchase_year,driven,fuel_type]).reshape(1, 5)))
            
            print(prediction)

            

            Dtime = datetime.now()

            # 6a0dcc9c03574ae0b563ea938316d37f
            random_string = uuid.uuid4().hex
            path = "static/" + random_string + ".svg"
            # model = load('mLinear.joblib')

            # CREATE TABLE multi (
            #     MultiID serial primary key NOT NULL,
            #     company_name text not null ,
            #     car_models text not null ,
            #     purchase_year int not null ,
            #     fuel_type text not null ,
            #     kilo_driven float not null,
            #     predict_price float not null,
            #     datetime text not null,
            #     image text,
            #     is_delete bool
            # );

            try:
                cur.execute("INSERT INTO multi(company_name,car_models,purchase_year,fuel_type,kilo_driven,predict_price,datetime,image,is_delete)VALUES (%s,%s,%d,%s,%f,%f,%s,%s,FALSE)", (company_name,car_model,purchase_year,fuel_type,driven,prediction,Dtime,random_string))
            except:
                return "Third"

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            return str(np.round(prediction[0],2))

            # if len(np_arr):
            #     make_picture('AgesAndHeights.pkl', model, np_arr, path)
            #     return render_template('mLinear.html', href=path)
            # else:
            #     return "Please enter Valid values => " + text

    except:
        return "Connection Fail"