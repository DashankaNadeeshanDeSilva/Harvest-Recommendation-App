# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import pickle

# Loading crop recommendation model
harvest_recommendation_model = pickle.load(open('models/RandomForest.pkl', 'rb'))

# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)

# render home page
@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('harvest_recommend.html', title=title)


@ app.route('/harvest_recommend_result', methods=['POST'])
def harvest_recommend_result():

    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosphorous'])
    K = int(request.form['Pottasium'])
    ph = float(request.form['PH'])
    rainfall = float(request.form['Rainfall'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = harvest_recommendation_model.predict(data)

    return render_template('harvest_recommend_result.html', prediction=prediction[0])



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
