from logging import exception
from flask import Flask, Blueprint, render_template, request, redirect, url_for
import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
from datetime import date, datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

hierarchical_algo = Blueprint("hierarchical_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")
# sLinear.secret_key = "cairocoders-ednalan"

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@hierarchical_algo.route('/hierarchical', methods=['GET', 'POST'])
def hierarchical():   
    try:
        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        if request_type_str == 'GET':
            return render_template('hierarchical_folder/hierarchical.html')
        else:
            fresh = request.form['fresh']
            milk = request.form['milk']
            grocery = request.form['grocery']
            frozen = request.form['frozen']
            paper = request.form['paper']
            delicassen = request.form['delicassen']
            
            ar = [] 
            ar.append(fresh)
            ar.append(milk)
            ar.append(grocery)
            ar.append(frozen)
            ar.append(paper)
            ar.append(delicassen)

            if ar:
                st = check_values(ar)

                if st=="Fail":
                    return "Write Posible values"
                
            Dtime = datetime.now()

            # CREATE TABLE hierarchical (
            #     hierarchicalID serial primary key NOT NULL,
            #     fresh int not null ,
            #     milk int not null ,
            #     grocery int not null ,
            #     frozen int not null ,
            #     paper int not null ,
            #     delicassen int not null ,
            #     predicted_value text not null ,
            #     datetime text not null,
            #     is_delete bool
            # );

            try:
                prediction = makemodel(ar)
            except Exception as e:
                print(e)

            if str(prediction) == '1':
                pred = "Retails"
            elif str(prediction) == '0':
                pred = "Hotel/Restaurant/Cafe"
            
            try:
                cur.execute("INSERT INTO hierarchical(fresh,milk,grocery,frozen,paper,delicassen,predicted_value,datetime,is_delete) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE)", (str(int(fresh)),str(int(milk)),str(int(grocery)),str(int(frozen)),str(int(paper)),str(int(delicassen)),str(pred),Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"

            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute('SELECT * FROM hierarchical ORDER BY datetime DESC')
            
            data = cur.fetchone()
            cur.close()
            print(data)

            return render_template('hierarchical_folder/hierarchical_pred.html', hierarchical = data)
            
    except Exception as e:
        print(e)

@hierarchical_algo.route('/hierarchical_layout', methods=['GET', 'POST'])
def hierarchical_layout():
    try: 
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM hierarchical WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('hierarchical_folder/hierarchical_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@hierarchical_algo.route('/hierarchical_delete/<string:id>', methods = ['POST','GET'])
def hierarchical_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE hierarchical SET is_delete = TRUE WHERE hierarchicalid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('hierarchical_algo.hierarchical_layout'))
    except:
        return "Connection Fail"

@hierarchical_algo.route('/hierarchical_edit/<string:id>', methods = ['POST', 'GET'])
def hierarchical_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM hierarchical WHERE hierarchicalid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()

        return render_template('hierarchical_folder/hierarchical_edit.html', hierarchical = data)
    except:
        return "Connection Fail"

@hierarchical_algo.route('/hierarchical_update/<string:id>', methods=['POST'])
def hierarchical_update(id):
    if request.method == 'POST':  
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        fresh = request.form['fresh']
        milk = request.form['milk']
        grocery = request.form['grocery']
        frozen = request.form['frozen']
        paper = request.form['paper']
        delicassen = request.form['delicassen']
        
        ar = [] 
        ar.append(fresh)
        ar.append(milk)
        ar.append(grocery)
        ar.append(frozen)
        ar.append(paper)
        ar.append(delicassen)

        if ar:
            st = check_values(ar)

            if st=="Fail":
                return "Write Posible values"
            
        Dtime = datetime.now()

        try:
            prediction = makemodel(ar)
        except Exception as e:
            print(e)

        if str(prediction) == '1':
            pred = "Retails"
        elif str(prediction) == '0':
            pred = "Hotel/Restaurant/Cafe"
        
        try:
            cur.execute("UPDATE hierarchical SET fresh = %s, milk = %s, grocery = %s, frozen = %s, paper = %s, delicassen = %s, predicted_value = %s, datetime = %s WHERE hierarchicalid = %s", (str(int(fresh)),str(int(milk)),str(int(grocery)),str(int(frozen)),str(int(paper)),str(int(delicassen)),str(pred),Dtime,id))
        except Exception as e:
            print(e)

        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cur.execute('SELECT * FROM hierarchical ORDER BY datetime DESC')
        
        data = cur.fetchone()
        cur.close()
        print(data)

        return render_template('hierarchical_folder/hierarchical_pred.html', hierarchical = data)
        

def check_values(values_array):
    ar = []
    for i in values_array:
        if float(i) and float(i) > 0:
            pass
        else:
            return "Fail"

    return "Success"

def makemodel(x_new):
    data = pd.read_csv('csv_files/Wholesale customers data.csv')
    X = data.drop(['Channel','Region'],axis=1)

    combined_array = np.append(X, x_new)
    combined_array = combined_array.reshape(441,6)

    X_scaled = normalize(combined_array)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
    
    return model.fit_predict(X_scaled)[-1]

