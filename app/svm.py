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


svm_algo = Blueprint("svm_algo", __name__, static_url_path = "", static_folder = "static", template_folder = "templates")

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@svm_algo.route('/svm', methods=['GET', 'POST'])
@cross_origin()
def svm():
    try:
        model = load('joblib_files/svm.joblib')

        request_type_str = request.method
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"

        if request_type_str == 'GET':
            try:
                return render_template('svm/svm.html', seq = [1,2,3,4,5,6,7,8,9,10])
            except Exception as e:
                print(e)
        else:
            clump = request.form['clump']
            unif_size = request.form['unif_size']
            unif_shape = request.form['unif_shape']
            marg_adh = request.form['marg_adh']
            sing_epi_size = request.form['sing_epi_size']
            bare_nuc = request.form['bare_nuc']
            bland_chrom = request.form['bland_chrom']
            norm_nucl = request.form['norm_nucl']
            mit = request.form['mit']
            
            x_real = np.array([int(clump), int(unif_size), int(unif_shape), int(marg_adh), int(sing_epi_size), int(bare_nuc), int(bland_chrom), int(norm_nucl), int(mit)]).reshape(1, -1)
            
            try:
                prediction = model.predict(pd.DataFrame(columns=['clump', 'unif_size', 'unif_shape', 'marg_adh', 'sing_epi_size', 'bare_nuc', 'bland_chrom', 'norm_nucl', 'mit'], data = x_real))
            except Exception as e:
                print(e)            
            
            Dtime = datetime.now()

            # CREATE TABLE svm (
            #     svmID serial primary key NOT NULL,
            #     clump int not null,
            #     unif_size int not null,
            #     unif_shape int not null,
            #     marg_adh int not null,
            #     sing_epi_size int not null,
            #     bare_nuc int not null,
            #     bland_chrom int not null,
            #     norm_nucl int not null,
            #     mit int not null,
            #     predict bool not null,
            #     datetime text not null,
            #     is_delete bool
            # );

            if int(prediction[0]) == 2:
                flag = "FALSE"
            else:
                flag = "TRUE"
            
            try:
                cur.execute("INSERT INTO svm(clump,unif_size,unif_shape,marg_adh,sing_epi_size,bare_nuc,bland_chrom,norm_nucl,mit,predict,datetime,is_delete)VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FALSE)", (int(clump),int(unif_size),int(unif_shape),int(marg_adh),int(sing_epi_size),int(bare_nuc),int(bland_chrom),int(norm_nucl),int(mit),flag,Dtime))
            except Exception as e:
                print(e)

            try:
                conn.commit()
                cur.close()
            except:
                return "Forth"
            
            try:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
                cur.execute('SELECT * FROM svm ORDER BY datetime DESC')
                
                data = cur.fetchone()
                cur.close()
                print(data)
                return render_template('svm/svm_pred.html', svm_data = data)
            except Exception as e:
                print(e)

    except Exception as e:
        print(e)

@svm_algo.route('/svm_layout', methods=['GET', 'POST'])
def svm_layout():
    try:      
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        except:
            return "Second"
        
        s = "SELECT * FROM svm WHERE is_delete='FALSE'"

        try:
            cur.execute(s)
        except Exception as e:
            print(e)
        list_users = cur.fetchall()
        cur.close()

        return render_template('svm/svm_layout.html', list_users = list_users)
    except Exception as e:
        print(e)


@svm_algo.route('/svm_delete/<string:id>', methods = ['POST','GET'])
def svm_delete(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('UPDATE svm SET is_delete = TRUE WHERE svmid = {0}'.format(id))
        conn.commit()
        cur.close()
        
        return redirect(url_for('svm_algo.svm_layout'))
    except:
        return "Connection Fail"

@svm_algo.route('/svm_edit/<string:id>', methods = ['POST', 'GET'])
def svm_edit(id):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute('SELECT * FROM svm WHERE svmid = %s', (id,))
        
        data = cur.fetchone()
        cur.close()
        
        return render_template('svm/svm_edit.html', svm_data = data, seq = [1,2,3,4,5,6,7,8,9,10])
    except Exception as e:
        print(e)

@svm_algo.route('/svm_update/<string:id>', methods=['POST'])
def svm_update(id):
    if request.method == 'POST':
        model = load('joblib_files/svm.joblib')

        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        clump = request.form['clump']
        unif_size = request.form['unif_size']
        unif_shape = request.form['unif_shape']
        marg_adh = request.form['marg_adh']
        sing_epi_size = request.form['sing_epi_size']
        bare_nuc = request.form['bare_nuc']
        bland_chrom = request.form['bland_chrom']
        norm_nucl = request.form['norm_nucl']
        mit = request.form['mit']
        
        x_real = np.array([int(clump), int(unif_size), int(unif_shape), int(marg_adh), int(sing_epi_size), int(bare_nuc), int(bland_chrom), int(norm_nucl), int(mit)]).reshape(1, -1)
        
        try:
            prediction = model.predict(pd.DataFrame(columns=['clump', 'unif_size', 'unif_shape', 'marg_adh', 'sing_epi_size', 'bare_nuc', 'bland_chrom', 'norm_nucl', 'mit'], data = x_real))
        except Exception as e:
            print(e)
        
        
        Dtime = datetime.now()

        if int(prediction[0]) == 2:
            flag = "FALSE"
        else:
            flag = "TRUE"
        
        try:
            cur.execute("UPDATE svm SET clump = %s, unif_size = %s, unif_shape = %s, marg_adh = %s, sing_epi_size = %s, bare_nuc = %s, bland_chrom = %s, norm_nucl = %s, mit = %s, predict = %s, datetime = %s WHERE svmid = %s", (int(clump),int(unif_size),int(unif_shape),int(marg_adh),int(sing_epi_size),int(bare_nuc),int(bland_chrom),int(norm_nucl),int(mit),flag,Dtime,id))
        except Exception as e:
            print(e)

        try:
            conn.commit()
            cur.close()
        except:
            return "Forth"
        
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            cur.execute('SELECT * FROM svm ORDER BY datetime DESC')
            
            data = cur.fetchone()
            cur.close()
            print(data)
            return render_template('svm/svm_pred.html', svm_data = data)
        except Exception as e:
            print(e)    
