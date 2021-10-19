from logging import exception
from os import name
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
from datetime import date, datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler


decision_tree_algo = Blueprint("decision_tree_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@decision_tree_algo.route('/decision_tree', methods=['GET', 'POST'])
@cross_origin()
def decision_tree():
    try:
        model = load('joblib_files/decision.joblib')
        titanic_data = pd.read_csv("csv_files/titanic.csv")

        titanic_data.Embarked = titanic_data.Embarked.fillna(titanic_data['Embarked'].mode()[0])
        median_age = titanic_data.Age.median()
        titanic_data.Age.fillna(median_age, inplace = True)
        titanic_data.drop('Cabin', axis = 1,inplace = True)
        titanic_data['Fare'] = titanic_data['Fare'].replace(0,titanic_data['Fare'].median())
        titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
        titanic_data['GenderClass'] = titanic_data.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)
        titanic_data = pd.get_dummies(titanic_data, columns=['GenderClass','Embarked'], drop_first=True)
        titanic = titanic_data.drop(['Name','Ticket','Sex','SibSp','Parch','PassengerId'], axis = 1)

        X = titanic.loc[:,titanic.columns != 'Survived']
        y = titanic.Survived 

        scaler = StandardScaler()

        scaler.fit(X)

        X = scaler.transform(X)

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'GET':
            try:
                return render_template('decision_tree/decision.html')
            except Exception as e:
                print(e)
        else:
            name = request.form['name']
            ticket_no = request.form['ticket_no']
            tclass_str = request.form['tclass']
            age = request.form['age']
            fare = request.form['fare']
            gender = request.form['gender']
            embarked = request.form['embarked']
            siblings = request.form['siblings']
            parents = request.form['parents']

            ar = [] 
            ar.append(int(age))
            ar.append(float(fare))
            ar.append(int(siblings))
            ar.append(int(parents))

            print(ar)
            if ar:
                st = check_values(ar)

                if st=="Success":
                    pass
                else:
                    return "Write Posible values!!!"

                if float(age) > 100:
                    return "Write Posible values"
            else:
                return "Write values"

            if gender == 'Male':
                female,male = 0,1
            else:
                female,male = 1,0

            if embarked == 'Queenstown':
                embarked_q,embarked_s = 1,0
            elif embarked == 'Southampton':
                embarked_q,embarked_s = 0,1
            else:
                embarked_q,embarked_s = 0,0

            if tclass_str == '1st':
                tclass = 1
            elif tclass_str == '2nd':
                tclass = 2
            else:
                tclass = 3

            family_size = int(siblings) + int(parents) + 1

            x_real = np.array([tclass,age,fare,family_size,female,male,embarked_q,embarked_s]).reshape(1, -1)
            x_real_process = scaler.transform(x_real)

            try:
                prediction = model.predict(pd.DataFrame(columns=['tclass','age','fare','family_size','female','male','embarked_q','embarked_s'], data = x_real_process))
            except Exception as e:
                print(e)
            
            Dtime = datetime.now()

            # CREATE TABLE decision (
            #     DecisionID serial primary key NOT NULL,
            #     t_name text not null,
            #     ticket_no text not null,
            #     tclass text not null,
            #     t_age int not null,
            #     fare float not null,
            #     gender text not null,
            #     embarked text not null,
            #     family_member int not null,
            #     predict int not null,
            #     datetime text not null,
            #     is_delete bool
            # );

            try:
                cur.execute("INSERT INTO decision(t_name,ticket_no,tclass,t_age,fare,gender,embarked,family_member,predict,datetime,is_delete)VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE)", (name,ticket_no,tclass,int(age),float(fare),gender,embarked,int(family_size),int(prediction[0]),Dtime))
            except Exception as e:
                print(e)
                
            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"
            
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM decision ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()

                return render_template('decision_tree/decision_pred.html', decision_tree = data)
            except Exception as e:
                print(e)

    except Exception as e:
        print(e)

@decision_tree_algo.route('/decision_layout', methods=['GET', 'POST'])
def decision_layout():
    try:      
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM decision WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('decision_tree/decision_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@decision_tree_algo.route('/decision_delete/<string:id>', methods = ['POST','GET'])
def decision_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE decision SET is_delete = TRUE WHERE decisionid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('decision_tree_algo.decision_layout'))
    except:
        return "Connection Fail"

@decision_tree_algo.route('/decision_edit/<string:id>', methods = ['POST', 'GET'])
def decision_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM decision WHERE decisionid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()
        
        return render_template('decision_tree/decision_edit.html', decision_tree_data = data)
    except Exception as e:
        print(e)

@decision_tree_algo.route('/decision_update/<string:id>', methods=['POST'])
def decision_update(id):
    
    if request.method == 'POST':        
        model = load('joblib_files/decision.joblib')
        titanic_data = pd.read_csv("csv_files/titanic.csv")
        
        titanic_data.Embarked = titanic_data.Embarked.fillna(titanic_data['Embarked'].mode()[0])
        median_age = titanic_data.Age.median()
        titanic_data.Age.fillna(median_age, inplace = True)
        titanic_data.drop('Cabin', axis = 1,inplace = True)
        titanic_data['Fare'] = titanic_data['Fare'].replace(0,titanic_data['Fare'].median())
        titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1
        titanic_data['GenderClass'] = titanic_data.apply(lambda x: 'child' if x['Age'] < 15 else x['Sex'],axis=1)
        titanic_data = pd.get_dummies(titanic_data, columns=['GenderClass','Embarked'], drop_first=True)
        titanic = titanic_data.drop(['Name','Ticket','Sex','SibSp','Parch','PassengerId'], axis = 1)

        X = titanic.loc[:,titanic.columns != 'Survived']
        y = titanic.Survived 

        scaler = StandardScaler()

        scaler.fit(X)

        X = scaler.transform(X)

        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
    
        name = request.form['name']
        ticket_no = request.form['ticket_no']
        tclass_str = request.form['tclass']
        age = request.form['age']
        fare = request.form['fare']
        gender = request.form['gender']
        embarked = request.form['embarked']
        family_member = request.form['family_member']  

        ar = [] 
        ar.append(int(age))
        ar.append(float(fare))
        ar.append(int(family_member))

        print(ar)
        if ar:
            st = check_values(ar)

            if st=="Success":
                pass
            else:
                return "Write Posible values!!!"

            if float(age) > 100:
                return "Write Posible values"
        else:
            return "Write values"

        if gender == 'Male':
            female,male = 0,1
        else:
            female,male = 1,0

        if embarked == 'Queenstown':
            embarked_q,embarked_s = 1,0
        elif embarked == 'Southampton':
            embarked_q,embarked_s = 0,1
        else:
            embarked_q,embarked_s = 0,0

        if tclass_str == '1st':
            tclass = 1
        elif tclass_str == '2nd':
            tclass = 2
        else:
            tclass = 3

        x_real = np.array([tclass,age,fare,family_member,female,male,embarked_q,embarked_s]).reshape(1, -1)
        x_real_process = scaler.transform(x_real)

        try:
            prediction = model.predict(pd.DataFrame(columns=['tclass','age','fare','family_size','female','male','embarked_q','embarked_s'], data = x_real_process))
        except Exception as e:
            print(e)
        
        Dtime = datetime.now()

        try:
            cur.execute("UPDATE decision SET t_name = %s, ticket_no = %s, tclass = %s,t_age = %s,fare = %s,gender = %s, embarked = %s, family_member = %s,predict = %s,datetime = %s WHERE decisionid = %s", (name,ticket_no,tclass,int(age),float(fare),gender,embarked,int(family_member),int(prediction[0]),Dtime,id))
        except Exception as e:
            print(e)
            
        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute('SELECT * FROM decision ORDER BY datetime DESC')
            
            data = cur.fetchone()
            cur.close()

            return render_template('decision_tree/decision_pred.html', decision_tree = data)
        except Exception as e:
            print(e)


def check_values(values_array):
    for i in values_array:
        # print("a ", float(i))
        if float(i) or int(i) == 0:
            # print(float(i))
            if float(i) > 0 or int(i) == 0:
                pass
            else:
                return "Fail"
        else:
            return "Fail"

    return "Success"