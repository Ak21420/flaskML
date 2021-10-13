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

mLinear = Blueprint("mLinear", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@mLinear.route('/multi_linear', methods=['GET', 'POST'])
@cross_origin()
def multi_linear():
    try:
        model=pickle.load(open('joblib_files/mLinear.pkl','rb'))
        car=pd.read_csv('csv_files/Cleaned_Car_data.csv')

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'GET':
            old_companies = car['company']
            old_car_models = car['name']

            d = {}

            for i in range(old_companies.shape[0]):
                d[old_car_models[i]] = old_companies[i]

            companies = sorted(car['company'].unique())
            car_models = sorted(car['name'].unique())
            year = sorted(car['year'].unique(),reverse=True)
            fuel_type = car['fuel_type'].unique()

            companies.insert(0,'Select Company')
            try:
                return render_template('multi_linear_folder/mLinear.html',companies=companies,car_models=car_models,years=year,fuel_types=fuel_type, com_model = d)
            except Exception as e:
                print(e)
        else:
            company_name = request.form['company']
            car_model = request.form['car_models']
            purchase_year = request.form['year']
            fuel_type = request.form['fuel_type']
            driven = request.form['kilo_driven']
                      
            purchase_year_new = float(purchase_year)

            a_list = []
            if driven:
                a_list.append(driven)
                st = check_values(a_list)

                if st=="Success":
                    driven_new = float(driven)
                else:
                    return "Write Posible values"
            
            try:
                prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model,company_name,purchase_year_new,driven_new,fuel_type]).reshape(1, 5)))
            except Exception as e:
                print(e)
            
            prediction_new = float(prediction)
            Dtime = datetime.now()

            # CREATE TABLE multi (
            #     MultiID serial primary key NOT NULL,
            #     company_name text not null ,
            #     car_models text not null ,
            #     purchase_year int not null ,
            #     fuel_type text not null ,
            #     kilo_driven float not null,
            #     predict_price float not null,
            #     datetime text not null,
            #     is_delete bool
            # );

            prediction_new = round(prediction_new,2)

            try:
                cur.execute("INSERT INTO multi(company_name,car_models,purchase_year,fuel_type,kilo_driven,predict_price,datetime,is_delete)VALUES (%s,%s,%s,%s,%s,%s,%s,FALSE)", (company_name,car_model,purchase_year_new,fuel_type,driven_new,prediction_new,Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"
            
            try:
                print(str(np.round(prediction[0],2)))

                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM multi ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()

                data['predict_price'] = round(data['predict_price'],2)

                return render_template('multi_linear_folder/mLinear_pred.html', mLinear = data)
            except Exception as e:
                print(e)

    except:
        return "Connection Fail"

@mLinear.route('/multi_linear_layout', methods=['GET', 'POST'])
def multi_linear_layout():
    try:      
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM multi WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('multi_linear_folder/mLinear_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@mLinear.route('/multi_Linear_delete/<string:id>', methods = ['POST','GET'])
def multi_Linear_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE multi SET is_delete = TRUE WHERE multiid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('mLinear.multi_linear_layout'))
    except:
        return "Connection Fail"

@mLinear.route('/multi_Linear_edit/<string:id>', methods = ['POST', 'GET'])
def multi_Linear_edit(id):
    try:
        car=pd.read_csv('csv_files/Cleaned_Car_data.csv')
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM multi WHERE multiid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        year = sorted(car['year'].unique(),reverse=True)
        fuel_type = car['fuel_type'].unique()

        companies.insert(0,'Select Company')
        
        return render_template('multi_linear_folder/mLinear_edit.html', mLinear = data, companies=companies,car_models=car_models,years=year,fuel_types=fuel_type)
    except:
        return "Connection Fail"

@mLinear.route('/multi_Linear_update/<string:id>', methods=['POST'])
def multi_Linear_update(id):
    if request.method == 'POST':
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':
            company_name=request.form['company']
            car_model=request.form['car_models']
            purchase_year=request.form['year']
            fuel_type=request.form['fuel_type']
            driven=request.form['kilo_driven']
            
            purchase_year_new = float(purchase_year)

            a_list = []
            if driven:
                a_list.append(driven)
                st = check_values(a_list)

                if st=="Success":
                    driven_new = float(driven)
                else:
                    return "Write Posible values"
                 
            Dtime = datetime.now()

            model=pickle.load(open('joblib_files/mLinear.pkl','rb'))

            try:
                prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'], data=np.array([car_model,company_name,purchase_year_new,driven_new,fuel_type]).reshape(1, 5)))
            except Exception as e:
                print(e)
            
            prediction_new = round(float(prediction),2)

            try:
                cur.execute("UPDATE multi SET company_name = %s, car_models = %s, purchase_year = %s, fuel_type = %s, kilo_driven = %s, predict_price = %s, datetime = %s WHERE multiid = %s", (company_name,car_model,purchase_year_new,fuel_type,driven_new,prediction_new,Dtime,id))
            except:
                return "Third"

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            try:
                print(str(np.round(prediction[0],2)))

                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM multi ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()

                data['predict_price'] = round(data['predict_price'],2)

                return render_template('multi_linear_folder/mLinear_pred.html', mLinear = data)
            except Exception as e:
                print(e)


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