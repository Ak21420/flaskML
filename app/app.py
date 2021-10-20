import re
from flask import Flask, Blueprint, render_template, request, redirect, url_for, flash,Response
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
from sLinear import sLinear
from mLinear import mLinear
from knn import knn_algo
from logistic import logistic_algo
from naive_bayes import naive_bayes_algo
from decision import decision_tree_algo
from svm import svm_algo
from kmeans import kmeans_algo

app = Flask(__name__, static_url_path = "", static_folder = "static", template_folder = "templates")
app.secret_key = "cairocoders-ednalan"

app.register_blueprint(sLinear, url_prefix="")
app.register_blueprint(mLinear, url_prefix="")
app.register_blueprint(knn_algo, url_prefix="")
app.register_blueprint(logistic_algo, url_prefix="")
app.register_blueprint(naive_bayes_algo, url_prefix="")
app.register_blueprint(decision_tree_algo, url_prefix="")
app.register_blueprint(svm_algo, url_prefix="")
app.register_blueprint(kmeans_algo, url_prefix="")

DB_HOST = "localhost"
DB_NAME = "mlModels"
DB_USER = "admin"
DB_PASS = "admin"

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


@app.route('/')
def home():
   return render_template('index.html')