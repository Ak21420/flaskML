import re
from flask import Flask, render_template, request, redirect, url_for, flash,Response
from flask_cors import CORS,cross_origin
from numpy.lib.function_base import append
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


app = Flask(__name__, static_url_path = "", static_folder = "static", template_folder = "templates")
app.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@app.route('/')
def home():
   return render_template('index.html')


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
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM simple WHERE is_delete=FALSE"
        cur.execute(s)
        list_users = cur.fetchall()
        cur.close()

        return render_template('sLinear_layout.html', list_users = list_users)
    except:
        return "Connection Fail"

@app.route('/simple_Linear_delete/<string:id>', methods = ['POST','GET'])
def simple_Linear_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE simple SET is_delete = TRUE WHERE simpleid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('simple_linear_layout'))
    except:
        return "Connection Fail"

@app.route('/simple_Linear_edit/<string:id>', methods = ['POST', 'GET'])
def simple_Linear_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM simple WHERE simpleid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()
        
        return render_template('sLinear_edit.html', sLinear = data)
    except:
        return "Connection Fail"

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


@app.route('/multi_linear', methods=['GET', 'POST'])
@cross_origin()
def multi_linear():
    try:
        model=pickle.load(open('mLinear.pkl','rb'))
        car=pd.read_csv('Cleaned_Car_data.csv')

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
                return render_template('mLinear.html',companies=companies,car_models=car_models,years=year,fuel_types=fuel_type, com_model = d)
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

                return render_template('mLinear_pred.html', mLinear = data)
            except Exception as e:
                print(e)

    except:
        return "Connection Fail"

@app.route('/multi_linear_layout', methods=['GET', 'POST'])
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

        return render_template('mLinear_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@app.route('/multi_Linear_delete/<string:id>', methods = ['POST','GET'])
def multi_Linear_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE multi SET is_delete = TRUE WHERE multiid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('multi_linear_layout'))
    except:
        return "Connection Fail"

@app.route('/multi_Linear_edit/<string:id>', methods = ['POST', 'GET'])
def multi_Linear_edit(id):
    try:
        car=pd.read_csv('Cleaned_Car_data.csv')
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM multi WHERE multiid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        companies = sorted(car['company'].unique())
        car_models = sorted(car['name'].unique())
        year = sorted(car['year'].unique(),reverse=True)
        fuel_type = car['fuel_type'].unique()

        companies.insert(0,'Select Company')
        
        return render_template('mLinear_edit.html', mLinear = data, companies=companies,car_models=car_models,years=year,fuel_types=fuel_type)
    except:
        return "Connection Fail"

@app.route('/multi_Linear_update/<string:id>', methods=['POST'])
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

            model=pickle.load(open('mLinear.pkl','rb'))
            car=pd.read_csv('Cleaned_Car_data.csv')

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

                return render_template('mLinear_pred.html', mLinear = data)
            except Exception as e:
                print(e)


@app.route('/logistic', methods=['GET', 'POST'])
def logistic():   
    try:
        model = load('logistic.joblib')
        data_set= pd.read_csv('Social_Network_Ads.csv') 

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('logistic.html')
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

                if age > 100:
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

                return render_template('logistic_pred.html', logistic = data)
            else:
                return "Please enter Valid values"
    except:
        return "Connection Fail"

@app.route('/logistic_layout', methods=['GET', 'POST'])
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

        return render_template('logistic_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@app.route('/logistic_delete/<string:id>', methods = ['POST','GET'])
def logistic_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE logistic SET is_delete = TRUE WHERE logisticid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('logistic_layout'))
    except:
        return "Connection Fail"

@app.route('/logistic_edit/<string:id>', methods = ['POST', 'GET'])
def logistic_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM logistic WHERE logisticid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('logistic_edit.html', logistic = data)
    except:
        return "Connection Fail"

@app.route('/logistic_update/<string:id>', methods=['POST'])
def logistic_update(id):
    if request.method == 'POST':  
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':    
            model = load('logistic.joblib')
            data_set= pd.read_csv('Social_Network_Ads.csv') 

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

                if age > 100:
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

                return render_template('logistic_pred.html', logistic = data)
            else:
                return "Please enter Valid values"


@app.route('/knn', methods=['GET', 'POST'])
def knn():  
    try:
        model = load('Knn.joblib')
        data_set= pd.read_csv('iris.csv') 

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('knn.html')
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

                return render_template('knn_pred.html', knn = data)
            else:
                return "Please enter Valid values"
    except Exception as e:
        print(e)


@app.route('/knn_layout', methods=['GET', 'POST'])
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

        return render_template('knn_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@app.route('/knn_delete/<string:id>', methods = ['POST','GET'])
def knn_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE knn SET is_delete = TRUE WHERE knnid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('knn_layout'))
    except:
        return "Connection Fail"

@app.route('/knn_edit/<string:id>', methods = ['POST', 'GET'])
def knn_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM knn WHERE knnid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('knn_edit.html', knn = data)
    except:
        return "Connection Fail"

@app.route('/knn_update/<string:id>', methods=['POST'])
def knn_update(id):
    if request.method == 'POST':      
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'POST':    
            model = load('Knn.joblib')
            data_set= pd.read_csv('iris.csv') 
         
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

                return render_template('knn_pred.html', knn = data)
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