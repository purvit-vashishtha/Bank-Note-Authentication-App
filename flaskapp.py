#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 21:49:44 2021

@author: Purvit Vashishtha
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')                 # decorator
def Welcome():
    return "Welcome"
    
@app.route('/predict')
def Predict_Note_Authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    
    return "Predicted Class:" + str(prediction)

@app.route('/predict_file',methods=["POST"])
def Prediction_Test():
    df_test = pd.read_csv(request.files.get("file"))
    prediction1 = classifier.predict(df_test)
    
    return "Predicted Class Values for CSV is:" + str(list(prediction1))
    





if __name__=='__main__':
    app.run(debug=True)