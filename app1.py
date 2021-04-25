#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:33:02 2021

@author: Purvit Vashishtha
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)


@app.route('/')                 # decorator
def welcome():
    return "Welcome"


@app.route('/predict', methods=["Get"])
def predict_note_authentication():
    """Let's Authenticate Bank Note
    This is using docstrings for specifications
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
     
    responses:
          200:
              description: The Output values
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    
    return str(prediction)


@app.route('/predict_file', methods=["POST"])
def prediction_note_file():
    """
    Let's Authenticate Bank Note
    This is using docstrings for specifications
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    
    responses:
        200:
            description: the output values
              

    """
    
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction1 = classifier.predict(df_test)
    
    return str(list(prediction1))
    

if __name__=='__main__':
    app.run(debug=True)